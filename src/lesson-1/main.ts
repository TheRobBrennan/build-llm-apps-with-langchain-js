import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";
import { ChatOpenAI } from "npm:@langchain/openai";
import { HumanMessage } from "npm:@langchain/core@^0.1.12/messages";
import { ChatPromptTemplate } from "npm:@langchain/core@^0.1.12/prompts";

// --------------------------------------------------------------------------
// Load environment variables from our .env file
// --------------------------------------------------------------------------
const env = config();

// Define our model
const model = new ChatOpenAI({
  openAIApiKey: env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo-1106",
});
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Language model
// --------------------------------------------------------------------------
const response = await model.invoke([
  new HumanMessage("Tell me a joke."),
]);

console.log(
  `response: ${JSON.stringify(response, null, 2)}\n\n${response.content}\n`,
);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Prompt template - Example 1
// --------------------------------------------------------------------------
const prompt = ChatPromptTemplate.fromTemplate(
  `What are three good names for a company that makes {product}?`,
);

const promptResponse = await prompt.format({
  product: "colorful socks",
});

console.log(
  `promptResponse: ${JSON.stringify(promptResponse, null, 2)}\n`,
);

const promptResponseMessages = await prompt.formatMessages({
  product: "colorful socks",
});

console.log(
  `promptResponseMessages: ${
    JSON.stringify(promptResponseMessages, null, 2)
  }\n`,
);
// --------------------------------------------------------------------------
