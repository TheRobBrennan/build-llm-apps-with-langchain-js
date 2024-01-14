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

// Add history and context
import { MessagesPlaceholder } from "npm:@langchain/core@^0.1.12/prompts";
import { AIMessage, HumanMessage } from "npm:@langchain/core@^0.1.12/messages";

// --------------------------------------------------------------------------
// PREREQUISITE: Lesson 5 continues with the previous vectorstore and retrieval chain
//
// For brevity, we will be consolidating the previous code into this block.
// --------------------------------------------------------------------------
const embeddings = new OpenAIEmbeddings();

// Ingest our PDF document
const loader = new PDFLoader("src/lesson-5/data/machine-learning.pdf");
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
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// At this point, we have a list of documents that are relevant to our query.
// However, they are not in a human-readable format - the matching documents
// are in a format that our LLM can reason about (i.e. <doc>...</doc>).
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

// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Adding history
//
// GOAL: Make a new chain that will rephrase the question as a follow-up question.
// --------------------------------------------------------------------------
const REPHRASE_QUESTION_SYSTEM_TEMPLATE =
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human",
    "Rephrase the following question as a standalone question:\n{question}",
  ],
]);

const rephraseQuestionChain = RunnableSequence.from([
  rephraseQuestionChainPrompt,
  new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
  new StringOutputParser(),
]);

// Step 1 - Ask the original question
const originalQuestion = "What are the prerequisites for this course?";
const originalAnswer = await retrievalChain.invoke({
  question: originalQuestion,
});
console.log(
  `\n${originalQuestion}\n\n${originalAnswer}\n\n`,
);

// Step 2 - Rephrase the question with chat history and the original follow-up question from the user
const chatHistory = [
  new HumanMessage(originalQuestion),
  new AIMessage(originalAnswer),
];

const rephrasedQuestionResponse = await rephraseQuestionChain.invoke({
  question: "Can you list them in bullet point form?",
  history: chatHistory,
});
console.log(`\n${rephrasedQuestionResponse}\n\n`);
