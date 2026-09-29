import "dotenv/config";

import { z } from "zod";
import {
    AIMessage,
    ToolMessage,
} from "@langchain/core/messages";

import { tool } from "@langchain/core/tools";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

import {
    StateGraph,
    MessagesAnnotation,
    START,
    END,
} from "@langchain/langgraph";


const llm = new ChatGoogleGenerativeAI({
    model: "gemini-3.5-flash-lite",
    temperature: 0.7,
    apiKey: process.env.GEMINI_API_KEY,
});


const multiply = tool(
    async ({ first, second }) => {
        return first * second;
    },
    {
        name: "multiply",
        description: "Multiplies two numbers",
        schema: z.object({
            first: z.number(),
            second: z.number(),
        }),
    }
);


const divide = tool(
    async ({ first, second }) => {
        if (second === 0) {
            throw new Error("Cannot divide by zero");
        }

        return first / second;
    },
    {
        name: "divide",
        description: "Divides two numbers",
        schema: z.object({
            first: z.number(),
            second: z.number(),
        }),
    }
);


const add = tool(
    async ({ first, second }) => {
        return first + second;
    },
    {
        name: "add",
        description: "Adds two numbers",
        schema: z.object({
            first: z.number(),
            second: z.number(),
        }),
    }
);


const sub = tool(
    async ({ first, second }) => {
        return first - second;
    },
    {
        name: "sub",
        description: "Subtracts two numbers",
        schema: z.object({
            first: z.number(),
            second: z.number(),
        }),
    }
);


const tools = [multiply, divide, add, sub];

const toolsByName = Object.fromEntries(
    tools.map(tool => [tool.name, tool])
);

const llmWithTools = llm.bindTools(tools);


// -------------------------
// LLM NODE
// -------------------------

async function llmCall(state) {

    const response = await llmWithTools.invoke([
        {
            role: "system",
            content:
                "You are a helpful assistant that performs arithmetic operations using tools.",
        },
        ...state.messages,
    ]);

    return {
        messages: [response],
    };
}


// -------------------------
// TOOL NODE
// -------------------------

async function toolNode(state) {

    const result = [];

    const lastMessage = state.messages.at(-1);

    if (lastMessage?.tool_calls?.length) {

        for (const toolCall of lastMessage.tool_calls) {

            const selectedTool = toolsByName[toolCall.name];

            if (!selectedTool) {
                throw new Error(
                    `Tool ${toolCall.name} not found`
                );
            }

            const observation =
                await selectedTool.invoke(toolCall.args);

            result.push(
                new ToolMessage({
                    content: String(observation),
                    tool_call_id: toolCall.id,
                })
            );
        }
    }

    return {
        messages: result,
    };
}


// -------------------------
// ROUTER
// -------------------------

function shouldContinue(state) {

    const lastMessage = state.messages.at(-1);

    if (lastMessage?.tool_calls?.length) {
        return "tools";
    }

    return END;
}


// -------------------------
// GRAPH
// -------------------------

const workflow = new StateGraph(MessagesAnnotation)
    .addNode("llmCall", llmCall)
    .addNode("tools", toolNode)

    .addEdge(START, "llmCall")

    .addConditionalEdges(
        "llmCall",
        shouldContinue,
        {
            tools: "tools",
            [END]: END,
        }
    )

    .addEdge("tools", "llmCall");

const app = workflow.compile();


// -------------------------
// RUN
// -------------------------

const result = await app.invoke({
    messages: [
        {
            role: "user",
            content: "What is 20 multiplied by 5?",
        },
    ],
});

console.log(result.messages.at(-1)?.content);