import { config } from "https://deno.land/x/dotenv/mod.ts";

// Load .env file
const env = config()

console.log(`Hello, world! Your OPENAI_API_KEY is ${env.OPENAI_API_KEY}`)
