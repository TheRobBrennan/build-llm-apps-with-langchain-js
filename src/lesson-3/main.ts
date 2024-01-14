import "../loaders/env_loader.ts";
import { OpenAIEmbeddings } from "npm:@langchain/openai";
import { similarity } from "npm:ml-distance";

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
