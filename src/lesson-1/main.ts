import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";
import { ChatOpenAI } from "npm:@langchain/openai";
import { HumanMessage } from "npm:@langchain/core@^0.1.12/messages";
import { StringOutputParser } from "npm:@langchain/core@^0.1.12/output_parsers";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "npm:@langchain/core@^0.1.12/prompts";
import {RunnableSequence} from "npm:@langchain/core@^0.1.12/runnables";

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
  `response:${JSON.stringify(response, null, 2)}\n\n${response.content}\n`,
);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Prompt template - Example 1
// --------------------------------------------------------------------------
const prompt = ChatPromptTemplate.fromTemplate(
  // NOTE: The template below is a single string that accepts an inlined product string
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

// --------------------------------------------------------------------------
// Prompt template - Example 2 - Explicitly define system and human
// messages using prompt templates
// --------------------------------------------------------------------------
const promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "You are an expert at picking company names.",
  ),
  HumanMessagePromptTemplate.fromTemplate(
    "What are three good names for a company that makes {product}?",
  ),
]);

const promptFromMessagesResponse = await promptFromMessages.formatMessages({
  product: "shiny objects",
});

console.log(
  `promptFromMessagesResponse: ${
    JSON.stringify(promptFromMessagesResponse, null, 2)
  }\n`,
);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Prompt template - Example 2a - Use shorthand to define messages
// --------------------------------------------------------------------------
const promptFromMessagesShorthand = ChatPromptTemplate.fromMessages([
  ["system", "You are an expert at picking company names."],
  ["human", "What are three good names for a company that makes {product}?"],
]);

const promptFromMessagesShorthandResponse = await promptFromMessagesShorthand
  .formatMessages({
    product: "shiny objects",
  });

console.log(
  `promptFromMessagesShorthandResponse: ${
    JSON.stringify(promptFromMessagesShorthandResponse, null, 2)
  }\n`,
);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// LangChain Expression Language (LCEL)
// --------------------------------------------------------------------------
const chain = prompt.pipe(model);

const chainResponse = await chain.invoke({
  product: "colorful socks",
});

console.log(
  `chainResponse: ${JSON.stringify(chainResponse, null, 2)}\n`,
);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// LangChain Expression Language (LCEL) - Output parser - String response
// --------------------------------------------------------------------------
const outputParser = new StringOutputParser();

const nameGenerationChain = prompt.pipe(model).pipe(outputParser);

const nameGenerationChainResponse = await nameGenerationChain.invoke({
  product: "fancy cookies",
});

// NOTE: The output parser returns a string instead of an object like previous examples
console.log(
  `nameGenerationChainResponse:\n\n${nameGenerationChainResponse}\n`,
);
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// LangChain Expression Language (LCEL) - Output parser - String response
// as a runnable sequence
// --------------------------------------------------------------------------
const nameGenerationChainAsRunnableSequence = RunnableSequence.from([prompt, model, outputParser]);

const nameGenerationChainAsRunnableSequenceResponse = await nameGenerationChain.invoke({
  product: "fancy cookies"
});

// NOTE: The output parser returns a string instead of an object like previous examples
console.log(
  `nameGenerationChainAsRunnableSequenceResponse:\n\n${nameGenerationChainAsRunnableSequenceResponse}\n`,
);
// --------------------------------------------------------------------------
