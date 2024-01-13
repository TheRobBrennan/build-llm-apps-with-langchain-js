// Load data from a GitHub repository
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import ignore from "ignore"; // Peer dependency, used to support .gitignore syntax

// Load PDF from a local folder
// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

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
// Splitting
// --------------------------------------------------------------------------
