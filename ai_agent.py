import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transform_codebase import *

class Chat:

    def __init__(self, role, pipeline, useHistory: bool):
        self.role = role
        self.pipeline = pipeline
        self.useHistory = useHistory
        self.system = {'role': 'system', 'content': self.role}
        self.history = [
            self.system
        ]
        self.terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def set_prompt(self, new_query: str) -> str:
        if self.useHistory:
            self.history.append(
                {
                    'role': 'user',
                    'content': new_query
                }
            )
            prompt = self.pipeline.tokenizer.apply_chat_template(
                self.history,
                tokenize = False,
                add_generation_prompt = True
            )
            
        else:
            temp = [
                self.system,
                {
                    'role': 'user',
                    'content': new_query
                }
            ]
            prompt = self.pipeline.tokenizer.apply_chat_template(
                temp,
                tokenize = False,
                add_generation_prompt = True
            )

        return prompt
        
    def run_conversation(self):
        while True:
            query = input()
            if query == '':
                break
            
            prompt = self.set_prompt(query)
            sequences = self.pipeline(
                prompt,
                eos_token_id = self.terminators,
                do_sample = True
            )
            response = sequences[0].get('generated_text').replace(prompt, '')

            if self.useHistory:
                self.history.append(
                    {'role': 'assistant', 'content': response}
                )
    
            print(response)
    
    def ask(self, query) -> str:
        if query == '':
            return 'Please, provide a valid question!'
        prompt = self.set_prompt(query)
        sequences = self.pipeline(
            prompt,
            eos_token_id = self.terminators,
            do_sample = True
        )
        response = sequences[0].get('generated_text').replace(prompt, '')

        if self.useHistory:
            self.history.append(
                {'role': 'assistant', 'content': response}
            )
            
        return response



class CodeQAChain:
    
    def __init__(self, retriever, textGenerator, searchGenerator):
        self.generator = textGenerator
        self.searchGenerator = searchGenerator
        self.retriever = retriever
        self.queryRetriever = '''
            Given the above question, extract the name of the classes or features that the user is asking for. The name of the classes are in Portuguese BR.
            The feature must be in English and the class name in Portuguese BR. Your answer must be in the format:
            **Class:** <name of the class extracted>
            **Feature:** <feature mentioned by the user> 
            and so on.
        '''
        self.queryUsingContext = '''
            Answer the above question based on the below context: \n **Context:**\n\n
        '''

        

    def searchContext(self, query: str, searchQuery, mappingDocs):
        #return context given a question that uses context to be answered
        pages = set() 
        if searchQuery == None:
            searchQuery = [self.searchGenerator.ask(query + "\n" + self.queryRetriever)]
            
        context = ""
        for query in searchQuery:
            searchDocs = self.retriever.get_relevant_documents(query)

            for doc in searchDocs:
                pages.add(doc.page_content)

        for doc in pages:
            context +=  "".join(doc) + "\n\n"

        #     for doc in searchDocs:
        #         path = mappingDocs[
        #             doc.page_content
        #         ]
        #         pages.add(path)

        # context = ''
        # for path in pages:
        #     with open(path, 'r') as f:
        #         content = f.readlines()

        #     context += "\n\n" + "".join(content)

        return context

    def run(self, query: str, useContext: bool, searchQuery: str, mappingDocs) -> str:
        ctx = None

        if useContext:
            ctx = self.searchContext(query, searchQuery, mappingDocs)
            query = query + self.queryUsingContext + ctx #add context retrieved from query

        return {
            'response': self.generator.ask(query),
            'context': ctx
        }

        
class CodeRAG:
    
    def __init__(self, repo_path, to_path, language, embeddingModel, pipeline, role, documentationList):
        self.map_suffixes = {
            'python': ['.py'],
            'java': ['.java'],
            'text': ['.txt']
            }

        self.repo_path = repo_path
        self.to_path = to_path
        self.language = language
        self.pipeline = pipeline
        self.embeddings = HuggingFaceEmbeddings(
            model_name = embeddingModel,
            model_kwargs = {
                'device': 'cuda'
            },
            encode_kwargs = {
                'normalize_embeddings': False
                }
        )

        self.separators = [
            '**Attributes:**',
            '**Methods:**',
            '**Standalone Functions:**',
            '\n'
        ]

        self.mappingDocs = dict()

        self.textGenerator = Chat(role, pipeline, useHistory = False)
        self.searchGenerator = Chat(
            'You are an AI assistant. Your task is to generate search queries.',
            pipeline,
            useHistory = False
        )
        self.qa = self.constructChat(documentationList)

    def cloneRepo(self, repo_path: str, to_path: str):
        return Repo.clone_from(repo_path, to_path=to_path)


    def loadFileSystem(self, path: str, glob: str, suffixes: list):
        loader = GenericLoader.from_filesystem(
            path, 
            glob=glob,
            suffixes=suffixes,
        )

        return loader.load()

    def createParser(self, language: str):
        return LanguageParser(
            language=language,
            parser_threshold=500
        )

    def splitDocuments(self, document_store, language):

        splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer = self.pipeline.tokenizer, chunk_size=2000, chunk_overlap = 200)
        # splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON, chunk_size=2000, chunk_overlap=200)
        texts = splitter.split_documents(document_store)
        return texts

    def createDB(self, texts):
        return FAISS.from_documents(texts, self.embeddings)
        
    def createRetriever(self, db, top_k):
        return db.as_retriever(
            search_type='mmr',
            search_kwargs={'k': top_k}
        )

    def createLLM(self, pipeline):
        return HuggingFacePipeline(
            pipeline=pipeline,
            model_kwargs={'temperature': 0, 'max_length': 1024}
        )
    
    def createQAChain(self, retriever, textGenerator, searchGenerator):

        return CodeQAChain(
            retriever,
            textGenerator,
            searchGenerator
        )
    
    def constructChat(self, documentationList):
        # if os.path.exists(self.to_path) == False:
        #     repo = self.cloneRepo(self.repo_path, self.to_path)

        # files = listAllFiles(self.to_path, self.map_suffixes[self.language][0])

        # docs = getDocumentation(files, self.mappingDocs, self.separators)
        
        texts = self.splitDocuments(documentationList, self.language)
        db = self.createDB(texts)
        retriever = self.createRetriever(db, 2)

        qa_chain = self.createQAChain(retriever, self.textGenerator, self.searchGenerator)
        
        return qa_chain
        
    def ask(self, query: str, useContext: bool, searchQuery: str):

        return self.qa.run(
            query,
            useContext,
            searchQuery,
            self.mappingDocs
        )