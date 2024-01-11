import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";
import { ChatOpenAI } from "npm:@langchain/openai";
import { HumanMessage } from "npm:@langchain/core@^0.1.12/messages";

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
  `Response: ${JSON.stringify(response, null, 2)}\n\n${response.content}\n`,
);
// --------------------------------------------------------------------------
