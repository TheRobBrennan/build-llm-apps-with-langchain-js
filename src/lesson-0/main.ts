import "../loaders/env_loader.ts";
import { ChatOpenAI } from "npm:@langchain/openai";

// Define our model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
});

// TEST: Ask our model a question
const response = await model.invoke("What is Sploosh.AI?");
console.log(
  `Response: ${JSON.stringify(response, null, 2)}\n\n${response.content}\n`,
);
