import "../loaders/env_loader.ts";

// TEST: Print our environment variables
console.log(`OPENAI_API_KEY: ${Deno.env.get("OPENAI_API_KEY")}\n`);
