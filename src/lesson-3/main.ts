import "../loaders/env_loader.ts";
import { OpenAIEmbeddings } from "npm:@langchain/openai";

// TEST: Print our environment variables
console.log(`OPENAI_API_KEY: ${Deno.env.get("OPENAI_API_KEY")}\n\n`);

// Vectorstore ingestion
const embeddings = new OpenAIEmbeddings();

// What do our embeddings look like?
const embeddingsResponse = await embeddings.embedQuery(
  "This is some sample text",
);
console.log(embeddingsResponse);
