import os
import glob
from git import Repo
from typing import List
from langchain.docstore.document import Document

def cloneRepo(repo_path: str, to_path: str):
    return Repo.clone_from(repo_path, to_path=to_path)

def listAllFiles(project_name: str, extension: str):
    return [
        filename for filename in glob.glob(project_name + '/**/*' + extension, recursive = True)
    ]

def file2txt(file_path: str) -> str:
    txt = ''
    with open(file_path, 'r') as f:
        for line in f:
            txt = "\n".join([txt, line])

    return txt

import uuid
import os

def convertCodeBase2Diagrams(function, agent):
    
    query = '''
    Explain this code: \n
    '''

    codeBrief = agent.ask(query + function) #agent must not use history
    
    folder = '.docs'
    extension = '.txt'
    new_path = str(uuid.uuid4()) + extension
    new_path = os.path.join(folder, new_path)
    print(new_path)
    with open(new_path, 'w') as f:
        f.write(codeBrief)


def createClassesList(paths: str, to_path: str):
    classNames = []
    for path in paths:
        with open(path, 'r') as f:
            classNames.append(f.readline()) #the first line always contains the class name
    
    with open(to_path, 'w') as f:
        f.writelines(classNames)

def getClassesList(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [string.replace('\n', '') for string in f.readlines()]

def getDocumentation(paths, mappingDocs, separators):
    documents = []
    for path in paths:
        
        with open(path, 'r') as f:
            content = f.readlines()
            
            for c in content:
                c = c.replace('\n', '')
                if c not in separators:
                    mappingDocs[c] = path

                    documents.append(Document(
                        page_content = c,
                        metadata = {'source': 'local'}
                    ))                    
                
    return documents

def getDocumentationFromString(func_string):
    
    return Document(
        page_content = func_string,
        metadata = {'source': 'local'}
    )        