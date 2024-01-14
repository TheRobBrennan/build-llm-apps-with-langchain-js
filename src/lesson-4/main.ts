import "../loaders/env_loader.ts";

// Vector embeddings and similarity
import { OpenAIEmbeddings } from "npm:@langchain/openai";

// Import and process a document
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// In-memory vectorstore
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Build our chain
import { RunnableSequence } from "npm:@langchain/core@^0.1.12/runnables";
import { Document } from "npm:@langchain/core@^0.1.12/documents";

// Synthesize a response
import { ChatPromptTemplate } from "npm:@langchain/core@^0.1.12/prompts";
import { RunnableMap } from "npm:@langchain/core@^0.1.12/runnables";

// Augmented generation
import { ChatOpenAI } from "npm:@langchain/openai";
import { StringOutputParser } from "npm:@langchain/core@^0.1.12/output_parsers";

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

// Helper function to convert documents to output our LLM can reason about
const convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`;
  }).join("\n");
};

/*
{
question: "What is deep learning?"
}
*/

// Build our chain
const documentRetrievalChain = RunnableSequence.from([
  (input) => input.question, // Step 1: Extract the question from the input
  retriever, // Step 2: Pass the question to the retriever
  convertDocsToString, // Step 3: Pipe documents from the retriever function to the helper function above
]);

// Run our chain
const results = await documentRetrievalChain.invoke({
  question: "What are the prerequisites for this course?",
});
console.log(results);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// At this point, we have a list of documents that are relevant to our query.
// However, they are not in a human-readable format - the matching documents
// are in a format that our LLM can reason about (i.e. <doc>...</doc>).
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Synthesize a response
// --------------------------------------------------------------------------
const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(
  TEMPLATE_STRING,
);

// NOTE: Our prompt requires an object with the context and question properties provided.
// We will use a runnable map for this - which calls all the runnables or runnable-like functions in parallel with the same output.
// The output is an object showing the results of each runnable.
const runnableMap = RunnableMap.from({
  context: documentRetrievalChain,
  // @ts-ignore - We don't need to be concerned with input types in this example
  question: (input) => input.question,
});

const runnableMapResponse = await runnableMap.invoke({
  question: "What are the prerequisites for this course?",
});

console.log(runnableMapResponse);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Augmented generation
// --------------------------------------------------------------------------

// Define our model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
});

// Build our retrieval chain
const retrievalChain = RunnableSequence.from([
  // Step 1: When an object is supplied to this initializer, it will be used as the input to the chain
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt, // Step 2: Pass the required input to the prompt (context and question)
  model, // Step 3: Pass the output to our model
  new StringOutputParser(),
]);

// Ask a question
const answer = await retrievalChain.invoke({
  question: "What are the prerequisites for this course?",
});

console.log(`\nAnswer: ${answer}\n`);
