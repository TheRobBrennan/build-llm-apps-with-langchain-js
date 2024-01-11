# Welcome

This project will explore working with LangChain.js - inspired by [DeepLearning.AI: Build LLM Apps with LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/).

## Getting started

This project will require an OpenAI API key as well as Deno.

Please see the _Install Deno_ section below if you do not have Deno installed on your system.

If you do not have an OpenAI account, please create a free account and a new API key for this project at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

Once you have an OpenAI API key that you can use for this project, please copy `.env.sample` to `.env` and replace the placeholder values.

If Deno has been installed on your system, you should be able to run the `main.js` file with:

```sh
% deno run --allow-read main.ts

# OPTIONAL: Run the start script in package.json
% npm start
```

### Install Deno

As part of the initial setup, please install Deno on your system.

```sh
# Install Deno on macOS
% curl -fsSL https://deno.land/install.sh | sh
```

Once Deno is installed, you may be encouraged to update your `$HOME/.zshrc` (or similar) file with something like:

```sh
# ~/.zshrc
...

export DENO_INSTALL="/Users/xxxxx/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"

```
