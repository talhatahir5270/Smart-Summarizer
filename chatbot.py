import re
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
import chromadb

# Initialize Chroma DB client
client = chromadb.PersistentClient(path='database')
my_collection = client.create_collection(
    name="collecti000",
    metadata={"hnsw:space": "cosine"}
)


class Chatbot:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", max_chunk_size=200):
        self.chunker = RecursiveChunker(model_name, max_chunk_size)
        self.model = "llama3.1:8b-instruct-q4_K_M"

    def process_document(self, text):
        self.chunker.chunk_text(text)
        self.chunker.embed_chunks()

    def query(self, user_query):
        query_embedding = self.chunker.embed_query(user_query)
        similar_chunks = self.chunker.search_similar_chunks(query_embedding, top_k=3)

        # Concatenate chunks into a single string
        result_string = "\n\n".join([chunk for chunk, _ in similar_chunks])

        # Use LLM to generate response
        system_part = f"You are an expert in Question and Answering. Answer the Question by Only using the text below. If the answer isn't present in the below Text, respond with 'I am sorry. I don't know the answer.***'\n{result_string}***"
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_part}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        conversation = [{"role": "assistant", "content": prompt}]
        response = ollama.chat(model=self.model, messages=conversation)

        return response['message']['content']

class RecursiveChunker:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", max_chunk_size=150):
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        self.chunks = []
        self.ids = []
        self.chunk_embeddings = []
        self.collection = my_collection
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text):
        paragraphs = text.split('\n\n')
        self.chunks = []

        for paragraph in paragraphs:
            if len(paragraph.split()) > self.max_chunk_size:
                indented_chunks = self.recursive_chunk_text(paragraph)
                self.chunks.extend(indented_chunks)
            else:
                self.chunks.append(paragraph)

    def recursive_chunk_text(self, text):
        words = text.split()
        if len(words) <= self.max_chunk_size:
            return [text]
        split_point = self.max_chunk_size
        while split_point > 0 and not words[split_point - 1].endswith(('.', '!', '?')):
            split_point -= 1
        if split_point == 0:
            split_point = self.max_chunk_size
        chunk = ' '.join(words[:split_point])
        remaining_text = ' '.join(words[split_point:])
        return [chunk] + self.recursive_chunk_text(remaining_text)

    def embed_chunks(self):
        self.chunk_embeddings = self.embeddings_model.embed_documents(self.chunks)
        self.ids = [f"idx{i+1}" for i in range(len(self.chunks))]
        self.collection.add(
            embeddings=self.chunk_embeddings,
            documents=self.chunks,
            ids=self.ids
        )

    def embed_query(self, query):
        return self.embeddings_model.embed_documents([query])[0]

    def search_similar_chunks(self, query_embedding, top_k=3):
        results = self.collection.query(query_embedding, n_results=top_k)
        similar_chunks = []
        for doc_list, score_list in zip(results['documents'], results['distances']):
            for doc, score in zip(doc_list, score_list):
                similar_chunks.append((doc, score))
        return similar_chunks
