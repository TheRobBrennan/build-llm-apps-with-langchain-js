import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";

import { ChatOpenAI } from "npm:@langchain/openai";

// Load environment variables from our .env file
const env = config();

// Define our model
const chatModel = new ChatOpenAI({
  openAIApiKey: env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo-1106",
});

// TEST: Ask our model a question
const response = await chatModel.invoke("What is Sploosh.AI?");
console.log(response.content);
