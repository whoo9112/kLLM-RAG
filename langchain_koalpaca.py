import torch 
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

loader = PyPDFLoader("./dataset_korean_news/manual_test2.pdf")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 150, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

# embed_model = SentenceTransformer('./jhgan-ko-sroberta-multitask/')
# embeddings = embed_model.encode(texts)
embeddings = HuggingFaceBgeEmbeddings(model_name ='jhgan/ko-sroberta-multitask')

docsearch = Chroma.from_documents(texts, embeddings) #, persist_directory="./dataset_korean_news"
# retriever = docsearch.as_retriever()



# llm = HuggingFacePipeline.from_model_id(model_id="")

model_id = "beomi/KoAlpaca-Polyglot-12.8B"
# model_id = "beomi/KoAlpaca-Polyglot-5.8B"

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    device = 0,
    task = "text-generation",
    model_kwargs = {
        "temperature": 0.7, 
        "max_length": 256,
        "torch_dtype": torch.float16
    },
)

embeddings_filter = EmbeddingsFilter(
    embeddings = embeddings, 
    similarity_threshold=0.9
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, 
    base_retriever=docsearch.as_retriever()
)

template = """
아래 주어진 context를 참조하여 답변합니다.

{context}

Question: {question}
답변: """

qa = RetrievalQA.from_chain_type(
    llm = llm, 
    chain_type="stuff", 
    retriever=compression_retriever, 
    return_source_documents =True, 
    verbose = True, 
    chain_type_kwargs={"prompt": PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )},
)

while(1):
    question= input("질문:")
    print(docsearch.similarity_search(question))
    res = qa({"query": f'{question}'})
    print(res)

