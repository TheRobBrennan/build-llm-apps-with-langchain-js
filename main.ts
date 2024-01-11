import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";

// Load .env file
const env = config()

// TODO: We're not going to be logging private keys in the future, but this is just for testing
console.log(`Hello, world! Your OPENAI_API_KEY is ${env.OPENAI_API_KEY}\n`)
