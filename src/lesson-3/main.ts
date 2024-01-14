import "../loaders/env_loader.ts";

// Vector embeddings and similarity
import { OpenAIEmbeddings } from "npm:@langchain/openai";
import { similarity } from "npm:ml-distance";

// Import and process a document
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// In-memory vectorstore
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// TEST: Print our environment variables
console.log(`OPENAI_API_KEY: ${Deno.env.get("OPENAI_API_KEY")}\n\n`);

// --------------------------------------------------------------------------
// Vectorstore ingestion
// --------------------------------------------------------------------------
const embeddings = new OpenAIEmbeddings();

// --------------------------------------------------------------------------
// EXAMPLE: What do our embeddings look like?
// --------------------------------------------------------------------------
const embeddingsResponse = await embeddings.embedQuery(
  "This is some sample text",
);
console.log(embeddingsResponse);

// --------------------------------------------------------------------------
// EXAMPLE: Let's compare vectors and see how similar they are
// --------------------------------------------------------------------------
const vector1 = await embeddings.embedQuery(
  "What are vectors useful for in machine learning?",
);

const unrelatedVector = await embeddings.embedQuery(
  "A group of parrots is called a pandemonium.",
);

const similarVector = await embeddings.embedQuery(
  "Vectors are representations of information.",
);

// --------------------------------------------------------------------------
// Calculate the cosine similarity between two vectors
// --------------------------------------------------------------------------
const unrelatedVectorSimilarityScore = similarity.cosine(
  vector1,
  unrelatedVector,
);
const relatedVectorSimilarityScore = similarity.cosine(vector1, similarVector);

console.log(
  "Unrelated vector similarity score: ",
  unrelatedVectorSimilarityScore,
);
console.log("Related vector similarity score: ", relatedVectorSimilarityScore);

// --------------------------------------------------------------------------
// EXAMPLE: Prepare a document for ingestion
// --------------------------------------------------------------------------
const loader = new PDFLoader("src/lesson-3/data/machine-learning.pdf");
const rawCS229Docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 128,
  chunkOverlap: 0,
});
const splitDocs = await splitter.splitDocuments(rawCS229Docs);

// For demo purposes, we are using an in-memory vectorstore initialized with our embeddings model
console.log("Initializing vectorstore...");
const vectorStore = new MemoryVectorStore(embeddings);

// Add our split documents to the vectorstore
console.log("Adding documents to vectorstore...");
await vectorStore.addDocuments(splitDocs);

// NOTE: At this point, we have a populated and searchable vectorstore 🤓

// EXERCISE: Try searching for a query in the vectorstore and return the top 4 results
console.log("Searching for query...");
const retrievedDocs = await vectorStore.similaritySearch(
  "What is deep learning?",
  4,
);
const pageContents = retrievedDocs.map((doc) => doc.pageContent);

// Display results
console.log(`Results: ${retrievedDocs.length} documents found!`);
console.log(pageContents);

// --------------------------------------------------------------------------
// Retrievers
// --------------------------------------------------------------------------
// Using our existing vectorstore, we can create a retriever that can be used
// as a LangChain Expresion Language (LCEL) runnable that can be used in a
// LangChain pipeline.
//
// Source - https://js.langchain.com/docs/expression_language/
// --------------------------------------------------------------------------
const retriever = vectorStore.asRetriever();
const retrieverResponse = await retriever.invoke("What is deep learning?");
console.log(retrieverResponse);
