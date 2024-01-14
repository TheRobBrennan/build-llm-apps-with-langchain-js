import "../loaders/env_loader.ts";

// Vector embeddings and similarity
import { OpenAIEmbeddings } from "npm:@langchain/openai";

// Import and process a document
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// In-memory vectorstore
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// --------------------------------------------------------------------------
// PREREQUISITE: Lesson 4 starts with the vectorstore from lesson 3 in a
// variety of helper functions that are not easily available. We can accomplish
// the same thing by replicating the vectorstore we created in lesson 3 directly.
//
// For brevity, we will be consolidating the lesson 3 vectorstore into this
// block.
// --------------------------------------------------------------------------
const embeddings = new OpenAIEmbeddings();

// Ingest our PDF document
const loader = new PDFLoader("src/lesson-4/data/machine-learning.pdf");
const rawCS229Docs = await loader.load();

// ENHANCEMENT: Let's use more production-ready chunking parameters
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1536,
  chunkOverlap: 128,
});
const splitDocs = await splitter.splitDocuments(rawCS229Docs);

// For demo purposes, we are using an in-memory vectorstore initialized with our embeddings model
console.log("Initializing vectorstore...");
const vectorStore = new MemoryVectorStore(embeddings);

// Add our split documents to the vectorstore
console.log("Adding documents to vectorstore...");
await vectorStore.addDocuments(splitDocs);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Build our retrieval chain
// --------------------------------------------------------------------------
const retriever = vectorStore.asRetriever();

// --------------------------------------------------------------------------
// Document retrieval in a chain
// --------------------------------------------------------------------------
