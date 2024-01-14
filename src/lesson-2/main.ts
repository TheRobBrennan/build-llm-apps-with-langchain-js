// Load data from a GitHub repository
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import ignore from "ignore"; // Peer dependency, used to support .gitignore syntax

// Load PDF from a local folder
// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

// Split our sources into chunks the LLM can use for reasoning
import {
  CharacterTextSplitter,
  RecursiveCharacterTextSplitter,
} from "langchain/text_splitter";

// --------------------------------------------------------------------------
// Load documents from a GitHub repository using a LangChain Document Loader
// --------------------------------------------------------------------------
const GITHUB_REPOSITORY_URL = "https://github.com/langchain-ai/langchainjs";
const GITHUB_IGNORED_PATHS = ["*.md", "yarn.lock"];

// Will not include anything under "ignorePaths" - and recursion is turned off for demo purposes
const loader = new GithubRepoLoader(
  GITHUB_REPOSITORY_URL,
  { recursive: false, ignorePaths: GITHUB_IGNORED_PATHS },
);

const docs = await loader.load();

console.log(docs.slice(0, 3));

// --------------------------------------------------------------------------
// Load PDF from a local folder using a LangChain PDF Loader
// --------------------------------------------------------------------------
const pdfLoader = new PDFLoader(
  "src/lesson-2/data/machine-learning.pdf",
);
const rawCS229Docs = await pdfLoader.load();

console.log(rawCS229Docs.slice(0, 5));

// --------------------------------------------------------------------------
// Splitting - Use JavaScript code to compare how splitting behaves
//
// NOTE: We are using a chunkSize that is smaller than one we would normally use
// --------------------------------------------------------------------------
const jsCodeSplitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 32,
  chunkOverlap: 0,
});

const jsCode = `function helloWorld() {
  console.log("Hello, World!");
  }
  // Call the function
  helloWorld();`;

const jsCodeSplitterResult = await jsCodeSplitter.splitText(jsCode);
console.log(
  `\nNaïve JavaScript code text splitter:\n${jsCodeSplitterResult}\n`,
);

// --------------------------------------------------------------------------
// Character Splitter - using an empty space as a delimiter
//
// NOTE: This is a naïve approach that would generate a result that would be
//      difficult to use for reasoning. It is included here for comparison only.
// --------------------------------------------------------------------------
const jsCodeCharacterSplitter = new CharacterTextSplitter({
  chunkSize: 32,
  chunkOverlap: 0,
  separator: " ",
});

const jsCodeCharacterSplitterResult = await jsCodeCharacterSplitter.splitText(
  jsCode,
);

console.log(
  `\nJavaScript code splitter by the empty space delimiter:\n${jsCodeCharacterSplitterResult}\n`,
);

// --------------------------------------------------------------------------
// RecursiveCharacterTextSplitter on the JavaScript code
//
// NOTE: This is a little less efficient, but it does a better job of splitting
// --------------------------------------------------------------------------
const jsCodeRecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
  .fromLanguage("js", { chunkSize: 64, chunkOverlap: 32 });

const jsCodeRecursiveCharacterTextSplitterResult =
  await jsCodeRecursiveCharacterTextSplitter.splitText(jsCode);

console.log(
  `\nJavaScript code splitter using overlapping chunks of a larger size:\n${jsCodeRecursiveCharacterTextSplitterResult}\n`,
);

// --------------------------------------------------------------------------
// RecursiveCharacterTextSplitter on the JavaScript code with larger chunks and overlap
//
// NOTE:
// --------------------------------------------------------------------------
const jsCodeRecursiveCharacterTextSplitterV2 = RecursiveCharacterTextSplitter
  .fromLanguage("js", { chunkSize: 512, chunkOverlap: 64 });

const jsCodeRecursiveCharacterTextSplitterV2Result =
  await jsCodeRecursiveCharacterTextSplitterV2.splitText(jsCode);

console.log(
  `\nJavaScript code splitter using larger chunk and overlap sizes:\n${jsCodeRecursiveCharacterTextSplitterV2Result}\n`,
);

// --------------------------------------------------------------------------
// Let's apply this approach to splitting our CS229 documents
// --------------------------------------------------------------------------
