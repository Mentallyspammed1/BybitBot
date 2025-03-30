#!/usr/bin/env node

// NeonCLI: An enhanced CLI tool for interacting with Google's Generative AI (Gemini) - Pyrmethus Edition - Ora-Free Version

const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
const dotenv = require('dotenv');
const fs = require('fs').promises;
const path = require('path');
const readline = require('readline');
require('colors'); // Switched to 'colors' - Pyrmethus Request
const mime = require('mime-types');
const yargs = require('yargs');
const { highlight } = require('cli-highlight');
// const ora = require('ora'); // Ora removed - Pyrmethus Request - No more ora!
const { spawn } = require('child_process');
const os = require('os');

// Load environment variables
dotenv.config();

// Constants
const APP_NAME = 'NeonCLI';
const APP_VERSION = '1.0.3'; // Pyrmethus Ora-Free Version Bump
const CONFIG_DIR = path.join(os.homedir(), '.neoncli');
const DEFAULT_CONFIG_PATH = path.join(CONFIG_DIR, 'config.json');
const DEFAULT_HISTORY_PATH = path.join(CONFIG_DIR, 'history.json');
const DEFAULT_MACROS_PATH = path.join(CONFIG_DIR, 'macros.json');
const DEFAULT_SESSIONS_DIR = path.join(CONFIG_DIR, 'sessions');
const DEFAULT_MODEL = 'gemini-1.5-flash-latest';
const DEFAULT_TEMPERATURE = 0.7;
const MAX_HISTORY_PAIRS = 100; // Increased History Limit - Pyrmethus Enhancement
const CMD_PREFIX = '/';
const MACRO_PREFIX = '!';
const TEMP_FILE_PREFIX = 'neoncli_temp_';
const EDITOR = process.env.EDITOR || 'nano';
const API_RETRY_LIMIT = 3; // Pyrmethus Retry Logic
const EXEC_TIMEOUT_MS = 30000; // Execution timeout constant - Pyrmethus Enhancement

// Default system prompt - Pyrmethus Refinement - More explicit security warning
const DEFAULT_SYSTEM_PROMPT_TEXT = `
You are ${APP_NAME} (v${APP_VERSION}), an advanced AI assistant running in a command-line interface.
Provide concise, accurate, and helpful responses. Use markdown for formatting where appropriate.

**IMPORTANT: Code Execution is EXPERIMENTAL and INSECURE.**
**Enable it ONLY if you understand the risks and in TRUSTED environments.**
**ALWAYS review code before execution. NeonCLI provides NO SANDBOXING.**

**Code Execution (If Enabled by User):**
You can request shell or Python code execution. The user MUST confirm each request.
Structure your request within a **single** JSON code block, preceded by "// EXECUTION REQUEST":
\`\`\`json
// EXECUTION REQUEST
{ "action": "run_shell", "command": "your_shell_command", "reason": "Explain why you need to run this." }
\`\`\`
Supported actions: "run_shell", "run_python", "save_shell", "save_python".
`.trim();

// Chalk styles - Pyrmethus Styling Polish - Added muted info style - Now using 'colors'
const neon = {
    separator: () => console.log('─'.repeat(80).grey), // .grey for gray in colors
    statusIdle: '● '.green,            // .green for green
    statusBusy: '● '.yellow,           // .yellow for yellow
    promptMarker: '>>'.cyan,          // .cyan for cyan
    pasteMarker: '[Paste]'.magenta,     // .magenta for magenta
    userMessage: (text) => text.blue,   // .blue for blue (using function for consistency, though direct string.blue also works)
    aiMessage: (text) => text.green,     // .green for green (using function for consistency)
    systemInfo: (text) => text.cyan,    // .cyan for cyan (using function for consistency)
    warning: (text) => text.yellow,    // .yellow for yellow (using function for consistency)
    error: (text) => text.red,        // .red for red (using function for consistency)
    debug: (text) => text.grey,        // .grey for gray in colors (using function for consistency)
    macroName: (text) => text.magenta,  // .magenta for magenta (using function for consistency)
    macroExpansion: (text) => text.magenta, // Removed .dim for colors - if dimming is needed, explore colors methods
    configKey: (text) => text.cyan,     // .cyan for cyan (using function for consistency)
    configValue: (text) => text.white,   // .white for white (using function for consistency)
    mutedInfo: (text) => '[INFO]'.grey.dim   // Muted info prefix style - Pyrmethus Enhancement - consistent prefix for muted logs
};

// Command-line arguments
const argv = yargs
    .option('api-key', { type: 'string', description: 'Google API Key' })
    .option('model', { type: 'string', description: 'Model name' })
    .option('temperature', { type: 'number', description: 'Temperature (0-1)' })
    .option('debug', { type: 'boolean', description: 'Enable debug mode' })
    .option('shell', { type: 'boolean', description: 'Enable shell execution' })
    .option('python', { type: 'boolean', description: 'Enable Python execution' })
    .help().argv;

// Utility Functions - Clamp function moved here to be defined before use - Pyrmethus Fix
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const logDebug = (...args) => { if (DEBUG_MODE) console.log(neon.debug, ...args); }; // Using neon.debug prefix - Pyrmethus Enhancement - Consistent debug logging
const logError = (...args) => console.log(neon.error, ...args); // Using neon.error prefix - Pyrmethus Enhancement - Consistent error logging
const logWarning = (...args) => console.log(neon.warning, ...args); // Using neon.warning prefix - Pyrmethus Enhancement - Consistent warning logging
const logSystem = (...args) => console.log(neon.systemInfo, ...args); // Using neon.systemInfo prefix - Pyrmethus Enhancement - Consistent system logging
const logMuted = (...args) => console.log(neon.mutedInfo, ...args); // Using neon.mutedInfo prefix - Pyrmethus Enhancement - Consistent muted logging
const logMacroExpansion = (name, args, result) => logDebug(`Macro !${name} "${args}" -> "${result}"`);
const clearConsole = () => process.stdout.write('\x1Bc');
const fileExists = async (filepath) => { try { await fs.access(filepath); return true; } catch { return false; } };
const ensureDirectoryExists = async (dir) => { if (!await fileExists(dir)) await fs.mkdir(dir, { recursive: true }); };


// Global state
let API_KEY = argv['api-key'] || process.env.GOOGLE_API_KEY;
let MODEL_NAME = argv.model || DEFAULT_MODEL;
let TEMPERATURE = clamp(argv.temperature || DEFAULT_TEMPERATURE, 0, 1); // clamp is now defined before use! - Pyrmethus Fix
let DEBUG_MODE = argv.debug || false;
let SHELL_ENABLED = argv.shell || false;
let PYTHON_ENABLED = argv.python || false;
let SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_TEXT;
let SAFETY_SETTINGS = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.MEDIUM },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.MEDIUM },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.MEDIUM },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.MEDIUM }
];
let HIGHLIGHT_ENABLED = true;
let HISTORY = [];
let macros = {};
let CURRENT_SESSION_ID = Date.now().toString();
let isPastingMode = false;
let pasteBuffer = [];
let isAiThinking = false;
let isProcessingMacro = false;
let isWaitingForShellConfirmation = false;
let isWaitingForPythonConfirmation = false;
let genAI, model, chatSession;
let readlineInterface;
const commandQueue = [];
let isProcessingQueue = false;

// Config Manager - Pyrmethus Enhancement - Added descriptions to config defaults
class ConfigManager {
    constructor(configPath) {
        this.configPath = configPath;
        this.config = {};
        this.defaults = {
            apiKey: null,
            model: DEFAULT_MODEL,
            temperature: DEFAULT_TEMPERATURE,
            debug: false,
            highlight: true,
            shellEnabled: false,
            pythonEnabled: false,
            systemPrompt: DEFAULT_SYSTEM_PROMPT_TEXT,
            historySize: MAX_HISTORY_PAIRS, // Added history size to config - Pyrmethus
        };
        this.descriptions = { // Descriptions for config keys - Pyrmethus
            apiKey: 'Google API Key',
            model: 'AI Model Name',
            temperature: 'AI Temperature (0-1)',
            debug: 'Enable Debug Mode (true/false)',
            highlight: 'Enable Syntax Highlighting (true/false)',
            shellEnabled: 'Enable Shell Execution (true/false) - DANGEROUS',
            pythonEnabled: 'Enable Python Execution (true/false) - DANGEROUS',
            systemPrompt: 'System Prompt for AI',
            historySize: `Maximum Chat History Size (pairs, currently ${MAX_HISTORY_PAIRS})` // Dynamic description
        };
    }

    async load() {
        await ensureDirectoryExists(CONFIG_DIR);
        try {
            const data = await fs.readFile(this.configPath, 'utf8');
            this.config = { ...this.defaults, ...JSON.parse(data) };
        } catch (e) {
            logMuted(`No config file found or invalid JSON at ${this.configPath}. Using defaults.`); // More informative muted log - Pyrmethus Enhancement
            this.config = { ...this.defaults };
            await this.save();
        }
        this.apply();
    }

    apply() {
        API_KEY = this.config.apiKey || API_KEY;
        MODEL_NAME = this.config.model || MODEL_NAME;
        TEMPERATURE = clamp(this.config.temperature || TEMPERATURE, 0, 1);
        DEBUG_MODE = this.config.debug || DEBUG_MODE;
        HIGHLIGHT_ENABLED = this.config.highlight !== false;
        SHELL_ENABLED = this.config.shellEnabled || SHELL_ENABLED;
        PYTHON_ENABLED = this.config.pythonEnabled || PYTHON_ENABLED;
        SYSTEM_PROMPT = this.config.systemPrompt || SYSTEM_PROMPT;
        // MAX_HISTORY_PAIRS = parseInt(this.config.historySize, 10) || MAX_HISTORY_PAIRS; // Not dynamically updating MAX_HISTORY_PAIRS after startup - complex to reload history cleanly.  Leaving as startup config for now.
    }

    async set(key, value) {
        if (key === 'temperature') value = clamp(parseFloat(value), 0, 1);
        else if (key === 'debug' || key === 'highlight' || key === 'shellEnabled' || key === 'pythonEnabled') {
            value = value.toLowerCase() === 'true' || value === '1' || value === true;
        } else if (key === 'historySize') { // Basic validation for historySize - Pyrmethus
            const intValue = parseInt(value, 10);
            if (isNaN(intValue) || intValue <= 0) {
                logWarning(`Invalid historySize. Must be a positive integer. Keeping current value.`);
                return;
            }
            value = intValue; // Store as integer in config
        } else if (!this.defaults.hasOwnProperty(key)) { // Validate config key - Pyrmethus Enhancement - Key validation on set
            return logWarning(`Unknown config key: ${key}. Use /config descriptions to see valid keys.`);
        }
        this.config[key] = value;
        this.apply();
        await this.save();
        await applyConfigChange(key);
    }

    getAll() { return { ...this.config }; }
    getDefaults() { return { ...this.defaults }; }
    getDescriptions() { return { ...this.descriptions }; } // Expose descriptions - Pyrmethus
    async save() {
        try {
            await fs.writeFile(this.configPath, JSON.stringify(this.config, null, 2), 'utf8');
            logDebug(`Config saved to ${this.configPath}`); // Debug log for config save - Pyrmethus Enhancement
        } catch (e) {
            logError(`Error saving config to ${this.configPath}:`, e.message); // More informative error log - Pyrmethus Enhancement
        }
    }
}

const configManager = new ConfigManager(DEFAULT_CONFIG_PATH);


const safePromptRefresh = () => {
    if (!readlineInterface || readlineInterface.closed) return;
    const status = isAiThinking ? neon.statusBusy : neon.statusIdle;
    const modelInfo = MODEL_NAME ? `(${MODEL_NAME})` : '';
    const pasteModeInfo = isPastingMode ? neon.pasteMarker : '';
    const waitingInfo = (isWaitingForShellConfirmation || isWaitingForPythonConfirmation) ? '[Confirm?] '.yellow : ''; // Using .yellow for colors
    const promptText = `${status}${waitingInfo}${neon.promptMarker}${modelInfo}${pasteModeInfo} `;
    readlineInterface.setPrompt(promptText);
    readlineInterface.prompt(true);
};

// Execution Functions
async function executeCommand(command, type, savePath = null) {
    const cmd = type === 'shell' ? command : 'python3';
    const args = type === 'shell' ? ['-c', command] : ['-c', command];
    const label = type === 'shell' ? 'Shell' : 'Python';
    // const spinner = ora(`Running ${label} command...`).start(); // Ora removed - Pyrmethus Request
    logSystem(`Running ${label} command...`); // Simple system log instead of spinner - Pyrmethus Request
    let output = null; // Initialize output to null - Pyrmethus Enhancement - clearer return value on failure
    try {
        output = await Promise.race([ // Pyrmethus Enhancement - Execution Timeout using Promise.race
            new Promise(async (resolve, reject) => {
                const proc = spawn(cmd, args, { shell: process.platform === 'win32' });
                let stdout = '', stderr = '';
                proc.stdout.on('data', (data) => stdout += data);
                proc.stderr.on('data', (data) => stderr += data);
                proc.on('close', (code) => {
                    if (code === 0) resolve(stdout || stderr); else reject(new Error(stderr || `Exited with code ${code}`));
                });
                proc.on('error', reject);
            }),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error(`${label} command timed out after ${EXEC_TIMEOUT_MS/1000} seconds`)), EXEC_TIMEOUT_MS) // Timeout error
            )
        ]);

        // spinner.succeed(`${label} command completed.`); // Ora removed - Pyrmethus Request
        logSystem(`${label} command completed.`); // Simple system log instead of spinner - Pyrmethus Request
        if (savePath) {
            await fs.writeFile(savePath, output, 'utf8');
            logMuted(`${label} output saved to ${savePath}`); // Muted log for save confirmation - Pyrmethus
        }
        return output;
    } catch (error) {
        // spinner.fail(`${label} command failed: ${error.message}`); // Ora removed - Pyrmethus Request
        logError(`${label} command failed: ${error.message}`); // Simple error log instead of spinner fail - Pyrmethus Request
        logError(`Execution Error (${label}):`, error.message); // More detailed error log - Pyrmethus
        return null; // Explicitly return null on error - Pyrmethus Enhancement - Consistent return value
    }
}

// History Management
async function loadHistory() {
    if (await fileExists(DEFAULT_HISTORY_PATH)) {
        try {
            const data = await fs.readFile(DEFAULT_HISTORY_PATH, 'utf8');
            HISTORY = JSON.parse(data).filter(entry => entry.role && Array.isArray(entry.parts));
            logMuted(`Chat history loaded from ${DEFAULT_HISTORY_PATH}`); // Muted log for history load - Pyrmethus
        } catch (e) {
            logWarning(`Warning: Could not load history from ${DEFAULT_HISTORY_PATH}. History may be corrupted or empty.`, e.message); // More informative warning - Pyrmethus
            HISTORY = []; // Ensure history is empty if load fails to avoid potential issues.
        }
    } else {
        logMuted(`No chat history file found at ${DEFAULT_HISTORY_PATH}. Starting fresh.`); // Muted log for no history file - Pyrmethus
    }
}


async function saveHistory() {
    await ensureDirectoryExists(CONFIG_DIR);
    const trimmed = HISTORY.slice(-MAX_HISTORY_PAIRS * 2);
    try {
        await fs.writeFile(DEFAULT_HISTORY_PATH, JSON.stringify(trimmed, null, 2), 'utf8');
        logDebug(`Chat history saved to ${DEFAULT_HISTORY_PATH}`); // Debug log for history save - Pyrmethus
    } catch (e) {
        logError(`Error saving chat history to ${DEFAULT_HISTORY_PATH}:`, e.message); // More informative error log - Pyrmethus
    }
}

async function clearHistory() { // Pyrmethus - Added clear history function
    HISTORY = [];
    await saveHistory();
    logSystem('Chat history cleared.');
}


// Macro Management
async function loadMacros() {
    if (await fileExists(DEFAULT_MACROS_PATH)) {
        try {
            const data = await fs.readFile(DEFAULT_MACROS_PATH, 'utf8');
            macros = JSON.parse(data);
            logMuted(`Macros loaded from ${DEFAULT_MACROS_PATH}`); // Muted log for macro load - Pyrmethus
        } catch (e) {
            logWarning(`Warning: Could not load macros from ${DEFAULT_MACROS_PATH}. Macros may be corrupted or file is invalid.`, e.message); // More informative warning - Pyrmethus
            macros = {}; // Ensure macros is empty if load fails.
        }
    } else {
        logMuted(`No macros file found at ${DEFAULT_MACROS_PATH}.`); // Muted log for no macros file - Pyrmethus
    }
}


async function saveMacros() {
    await ensureDirectoryExists(CONFIG_DIR);
    try {
        await fs.writeFile(DEFAULT_MACROS_PATH, JSON.stringify(macros, null, 2), 'utf8');
        logDebug(`Macros saved to ${DEFAULT_MACROS_PATH}`); // Debug log for macro save - Pyrmethus
    } catch (e) {
        logError(`Error saving macros to ${DEFAULT_MACROS_PATH}:`, e.message); // More informative error log - Pyrmethus
    }
}

function expandMacro(line, depth = 0) {
    const MAX_MACRO_DEPTH = 5; // Constant for max macro depth - Pyrmethus
    if (depth > MAX_MACRO_DEPTH) {
        logWarning(`Maximum macro expansion depth (${MAX_MACRO_DEPTH}) reached. Possible recursion loop.`);
        return null;
    }
    if (!line.startsWith(MACRO_PREFIX) || isProcessingMacro) return null;
    const match = line.match(/^!([a-zA-Z0-9_-]+)(?:\s+(.*))?$/s);
    if (!match) return null;
    const name = match[1];
    const argsString = match[2]?.trim() || '';
    if (macros.hasOwnProperty(name)) {
        isProcessingMacro = true;
        logDebug(`Expanding macro: !${name} with args: "${argsString}" (depth: ${depth})`);
        let expanded = macros[name];
        const argList = argsString.split(/\s+/).filter(Boolean);
        expanded = expanded.replace(/\$0/g, name);
        expanded = expanded.replace(/\$\*/g, argsString);
        expanded = expanded.replace(/\$#/g, argList.length.toString());
        argList.forEach((arg, index) => {
            const placeholder = new RegExp(`\\$${index + 1}`, 'g');
            expanded = expanded.replace(placeholder, arg);
        });
        if (expanded.startsWith(MACRO_PREFIX)) {
            const nestedExpanded = expandMacro(expanded, depth + 1);
            if (nestedExpanded !== null) expanded = nestedExpanded;
        }
        logMacroExpansion(name, argsString, expanded);
        isProcessingMacro = false;
        return expanded.trim();
    } else {
        logWarning(`Macro ${neon.macroName(`!${name}`)} not found. Interpreting literally.`);
        return null;
    }
}

// AI Interaction
async function sendMessageToAI(parts, retryCount = 0) { // Pyrmethus - Retry mechanism for sendMessageToAI
    if (!chatSession) {
        logError("AI model not initialized.");
        return;
    }
    isAiThinking = true;
    safePromptRefresh();
    // const spinner = ora('AI is thinking...').start(); // Ora removed - Pyrmethus Request
    logSystem('AI is thinking...'); // Simple system log instead of spinner - Pyrmethus Request
    let responseText = '';
    let aiResponseReceived = false; // Flag to track if AI response started - Pyrmethus Enhancement - Improved error handling in streaming

    try {
        const result = await chatSession.sendMessageStream(parts);
        // spinner.stop(); // Ora removed - Pyrmethus Request - No spinner stop needed
        process.stdout.write(neon.aiMessage('AI: '));
        aiResponseReceived = true; // AI response stream started successfully - Pyrmethus Enhancement
        for await (const chunk of result.stream) {
            const text = chunk.text();
            responseText += text;
            process.stdout.write(text);
        }
        console.log();
        HISTORY.push({ role: 'user', parts }, { role: 'model', parts: [{ text: responseText }] });
        await saveHistory();
        const execResult = await detectAndHandleExecutionRequest(responseText);
        if (execResult.handled && execResult.output) {
            await sendMessageToAI([{ text: `Execution output: ${execResult.output}` }]);
        }
    } catch (error) {
        // spinner.fail(`AI error: ${error.message}`); // Ora removed - Pyrmethus Request
        logError(`AI error: ${error.message}`); // Simple error log instead of spinner fail - Pyrmethus Request
        logError(`AI Interaction Error: ${error.message}`); // More specific error log - Pyrmethus
        if (!aiResponseReceived) { // Pyrmethus Enhancement - More informative error if stream never started
            logError("AI response stream failed to start. Check API key, model, and network.");
        }
        if (retryCount < API_RETRY_LIMIT) { // Retry logic - Pyrmethus
            const retryDelay = Math.pow(2, retryCount) * 1000; // Exponential backoff
            logWarning(`Retrying AI request in ${retryDelay/1000} seconds... (Attempt ${retryCount + 1}/${API_RETRY_LIMIT})`);
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            return sendMessageToAI(parts, retryCount + 1); // Recursive retry
        } else {
            logError(`AI request failed after ${API_RETRY_LIMIT} retries. Please check your API key and network connection.`);
        }
    } finally {
        isAiThinking = false;
        safePromptRefresh();
    }
}

async function detectAndHandleExecutionRequest(responseText) {
    const execRegex = /\/\/ EXECUTION REQUEST\s*```json\s*(\{[\s\S]+?\})\s*```/s;
    const match = responseText.match(execRegex);
    if (!match) return { handled: false };
    let request;
    try {
        request = JSON.parse(match[1]);
    } catch (e) {
        logError("Error parsing execution request JSON:", e.message); // More specific error log - Pyrmethus
        logDebug("Failed JSON:", match[1]); // Debug log of failed JSON - Pyrmethus
        return { handled: false };
    }
    const { action, command, reason, savePath } = request;
    if (!['run_shell', 'run_python', 'save_shell', 'save_python'].includes(action)) return { handled: false };
    const isShell = action.includes('shell');
    const isSave = action.includes('save');
    const enabled = isShell ? SHELL_ENABLED : PYTHON_ENABLED;
    if (!enabled) {
        logWarning(`${isShell ? 'Shell' : 'Python'} execution is disabled. Enable with /${isShell ? 'shell' : 'python'} on.`);
        return { handled: true };
    }
    logSystem(`AI requests ${isShell ? 'shell' : 'Python'} execution: "${command}"\nReason: ${reason}`);
    const confirm = isShell ? 'isWaitingForShellConfirmation' : 'isWaitingForPythonConfirmation';
    global[confirm] = true;
    safePromptRefresh();
    const response = await new Promise(resolve => {
        readlineInterface.question('Execute? (yes/no): ', answer => resolve(answer.trim().toLowerCase() === 'yes'));
    });
    global[confirm] = false;
    safePromptRefresh();
    if (!response) {
        logSystem('Execution declined.');
        return { handled: true };
    }
    const output = await executeCommand(command, isShell ? 'shell' : 'python', isSave ? savePath : null);
    return { handled: true, output };
}

// Command Handlers - Pyrmethus Command Enhancements and Additions
const commandHandlers = {
    help: async () => {
        neon.separator();
        logSystem(`**${APP_NAME} Commands:**
- /help - Show this help
- /exit | /quit | /bye - Exit the application
- /clear - Clear the screen
- /history - Show chat history
- /history clear - Clear chat history  ${neon.mutedInfo}
- /file <path> - Send a file to the AI
- /paste - Start multi-line input mode
- /endpaste - End multi-line input mode
- /save <path> - Save session to file
- /edit - Edit last message or system prompt
- /temp <value> - Set temperature (0-1)
- /model <name> - Set AI model
- /shell on|off|run|save - Manage shell execution (DANGEROUS, use with caution)
- /python on|off|run|save - Manage Python execution (DANGEROUS, use with caution)
- /macro <name> [value] - Set or show macro
- /macro list - List all macros  ${neon.mutedInfo}
- /config - Show current configuration
- /config set <key> <value> - Set configuration value
- /config reset <key> - Reset configuration key to default
- /config defaults - Show default configuration  ${neon.mutedInfo}
- /config descriptions - Show configuration descriptions ${neon.mutedInfo}
- /highlight on|off - Toggle syntax highlighting
- /debug on|off - Toggle debug mode

**Code Execution:**
- Enabling code execution allows the AI to request running code on your system.
- This is experimental and NOT SECURE. Use only in trusted environments.
- **SECURITY WARNING: NeonCLI provides NO SANDBOXING. Exercise extreme caution.** ${neon.warning}
- Always review the code before confirming execution.

**Macros:**
- Macros allow you to save and reuse commands or text snippets.
- Use '!' prefix to trigger macro expansion (e.g., !mymacro).
- Macros can accept arguments ($1, $2, $*, $#, $0).`);
        neon.separator();
    },
    exit: async () => gracefulExit(),
    quit: async () => gracefulExit(),
    bye: async () => gracefulExit(),
    clear: async () => clearConsole(),
    history: async (args) => {
        if (args.trim() === 'clear') { // Pyrmethus - Added history clear subcommand
            await clearHistory();
        } else {
            neon.separator();
            if (HISTORY.length === 0) {
                logMuted('Chat history is empty.'); // Muted message for empty history - Pyrmethus
            } else {
                HISTORY.forEach((entry, i) => {
                    const prefix = entry.role === 'user' ? neon.userMessage('You: ') : neon.aiMessage('AI: ');
                    console.log(`${prefix}${entry.parts.map(p => p.text || '[data]').join(' ')}`);
                });
            }
            neon.separator();
        }
    },
    file: async (args) => {
        const filepath = args.trim();
        if (!filepath) return logWarning('File path is required. Usage: /file <path>'); // Input validation - Pyrmethus
        if (!await fileExists(filepath)) return logError(`File not found: ${filepath}`);
        const mimeType = mime.lookup(filepath) || 'application/octet-stream';
        try {
            const data = await fs.readFile(filepath);
            await queueTask({ type: 'message', handler: sendMessageToAI, parts: [{ inlineData: { data: data.toString('base64'), mimeType }, filename: path.basename(filepath) }] }); // Added filename for context - Pyrmethus
            logMuted(`File "${path.basename(filepath)}" sent to AI.`); // Muted log for file sent - Pyrmethus
        } catch (e) {
            logError(`Error reading file "${filepath}":`, e.message); // More specific error log - Pyrmethus
        }

    },
    paste: async () => { isPastingMode = true; pasteBuffer = []; logSystem('Paste mode started. Use /endpaste to finish.'); safePromptRefresh(); },
    endpaste: async () => {
        if (!isPastingMode) return logWarning('Not in paste mode.');
        isPastingMode = false;
        const content = pasteBuffer.join('\n');
        pasteBuffer = [];
        if (content) await queueTask({ type: 'message', handler: sendMessageToAI, parts: [{ text: content }] });
        safePromptRefresh();
        logMuted('Paste mode ended.'); // Muted log for paste mode end - Pyrmethus
    },
    save: async (args) => {
        const filepath = path.resolve(args.trim() || path.join(DEFAULT_SESSIONS_DIR, `${CURRENT_SESSION_ID}.json`));
        await ensureDirectoryExists(path.dirname(filepath));
        try {
            await fs.writeFile(filepath, JSON.stringify(HISTORY, null, 2), 'utf8');
            logSystem(`Session saved to ${filepath}`);
        } catch (e) {
            logError(`Error saving session to ${filepath}:`, e.message); // More specific error log - Pyrmethus
        }

    },
    edit: async () => {
        const lastUser = HISTORY.slice().reverse().find(h => h.role === 'user');
        const content = lastUser ? lastUser.parts.map(p => p.text).join('\n') : SYSTEM_PROMPT;
        const updated = await openInEditor(content);
        if (updated && lastUser) {
            lastUser.parts = [{ text: updated }];
            await saveHistory();
            await sendMessageToAI([{ text: updated }]);
        } else if (updated) {
            SYSTEM_PROMPT = updated;
            await configManager.set('systemPrompt', updated);
        }
    },
    temp: async (args) => {
        const temp = parseFloat(args);
        if (isNaN(temp) || temp < 0 || temp > 1) return logWarning('Temperature must be a number between 0 and 1.'); // Input validation - Pyrmethus
        await configManager.set('temperature', temp);
        logSystem(`Temperature set to ${TEMPERATURE}`);
    },
    model: async (args) => {
        const name = args.trim();
        if (!name) return logWarning('Model name required. Usage: /model <model_name>'); // Input validation - Pyrmethus
        await configManager.set('model', name);
        logSystem(`Model set to ${MODEL_NAME}`);
    },
    shell: async (args) => {
        const [subcmd, ...rest] = args.trim().split(/\s+/);
        if (subcmd === 'on') {
            await configManager.set('shellEnabled', true);
            logWarning("WARNING: Enabling shell execution is DANGEROUS. Use with extreme caution and review all code."); // Stronger warning - Pyrmethus
            logSystem('Shell execution enabled.');
        } else if (subcmd === 'off') {
            await configManager.set('shellEnabled', false);
            logSystem('Shell execution disabled.');
        } else if (subcmd === 'run' && SHELL_ENABLED) {
            const output = await executeCommand(rest.join(' '), 'shell');
            if (output) console.log(output);
        } else if (subcmd === 'save' && SHELL_ENABLED) {
            const cmd = rest.slice(0, -1).join(' ');
            const filepath = rest[rest.length - 1];
            await executeCommand(cmd, 'shell', filepath);
            logSystem(`Output saved to ${filepath}`);
        } else {
            logWarning('Usage: /shell on|off|run|save');
        }
    },
    python: async (args) => {
        const [subcmd, ...rest] = args.trim().split(/\s+/);
        if (subcmd === 'on') {
            await configManager.set('pythonEnabled', true);
            logWarning("WARNING: Enabling Python execution is DANGEROUS. Use with extreme caution and review all code."); // Stronger warning - Pyrmethus
            logSystem('Python execution enabled.');
        } else if (subcmd === 'off') {
            await configManager.set('pythonEnabled', false);
            logSystem('Python execution disabled.');
        } else if (subcmd === 'run' && PYTHON_ENABLED) {
            const output = await executeCommand(rest.join(' '), 'python');
            if (output) console.log(output);
        } else if (subcmd === 'save' && PYTHON_ENABLED) {
            const cmd = rest.slice(0, -1).join(' ');
            const filepath = rest[rest.length - 1];
            await executeCommand(cmd, 'python', filepath);
            logSystem(`Output saved to ${filepath}`);
        } else {
            logWarning('Usage: /python on|off|run|save');
        }
    },
    macro: async (args) => {
        const [subcmd, ...macroArgs] = args.trim().split(/\s+/);
        if (subcmd === 'list') { // Pyrmethus - Added macro list subcommand
            const macroNames = Object.keys(macros);
            if (macroNames.length === 0) {
                logMuted('No macros defined.'); // Muted message for no macros - Pyrmethus
            } else {
                neon.separator();
                logSystem('Defined Macros:'.cyan); // Using .cyan for systemInfo style with colors
                macroNames.forEach(name => console.log(`- ${neon.macroName(`!${name}`)}: ${macros[name].substring(0, 50)}...`)); // Show macro name and first 50 chars of value
                neon.separator();
            }
        } else {
            const name = subcmd;
            const value = macroArgs.join(' ');
            if (!name) {
                Object.entries(macros).forEach(([k, v]) => console.log(`${neon.macroName(`!${k}`)}: ${v}`)); // Fallback to showing all macros if no name after /macro
            } else if (value) {
                macros[name] = value;
                await saveMacros();
                logSystem(`Macro ${neon.macroName(`!${name}`)} set to "${value}"`);
            } else if (macros[name]) {
                console.log(`${neon.macroName(`!${name}`)}: ${macros[name]}`);
            } else {
                logWarning(`Macro ${neon.macroName(`!${name}`)} not found.`);
            }
        }
    },

    config: async (args) => {
        const [action, key, ...valueParts] = args.trim().split(/\s+/);
        const value = valueParts.join(' ');
        if (!action) {
            const config = configManager.getAll();
            neon.separator();
            logSystem('Current Configuration:'.cyan); // Using .cyan for systemInfo style with colors
            Object.entries(config).forEach(([k, v]) => {
                console.log(`  ${neon.configKey(k)}: ${neon.configValue(v)}`);
            });
            neon.separator();
        } else if (action === 'set' && key && value) {
            if (!configManager.getDefaults().hasOwnProperty(key)) { // Validate config key - Pyrmethus
                return logWarning(`Unknown config key: ${key}. Use /config descriptions to see valid keys.`);
            }
            await configManager.set(key, value);
            logSystem(`Set ${neon.configKey(key)} to ${neon.configValue(value)}`);
        } else if (action === 'reset' && key) {
            if (!configManager.getDefaults().hasOwnProperty(key)) { // Validate config key - Pyrmethus
                return logWarning(`Unknown config key: ${key}. Use /config descriptions to see valid keys.`);
            }
            const defaults = configManager.getDefaults();
            if (defaults.hasOwnProperty(key)) {
                await configManager.set(key, defaults[key]);
                logSystem(`Reset ${neon.configKey(key)} to default: ${neon.configValue(defaults[key])}`);
            } else {
                logWarning(`Unknown config key: ${key}`); // Redundant check, but kept for clarity
            }
        } else if (action === 'defaults') { // Pyrmethus - Added config defaults subcommand
            const defaults = configManager.getDefaults();
            neon.separator();
            logSystem('Default Configuration:'.cyan); // Using .cyan for systemInfo style with colors
            Object.entries(defaults).forEach(([k, v]) => {
                console.log(`  ${neon.configKey(k)}: ${neon.configValue(v)}`);
            });
            neon.separator();

        } else if (action === 'descriptions') { // Pyrmethus - Added config descriptions subcommand
            const descriptions = configManager.getDescriptions();
            neon.separator();
            logSystem('Configuration Key Descriptions:'.cyan); // Using .cyan for systemInfo style with colors
            Object.entries(descriptions).forEach(([k, v]) => {
                console.log(`  ${neon.configKey(k)}: ${neon.configValue(v)}`);
            });
            neon.separator();
        }
         else {
            logWarning("Usage: /config [set <key> <value> | reset <key> | defaults | descriptions]"); // Updated help message - Pyrmethus
        }
    },
    highlight: async (args) => {
        const state = args.trim().toLowerCase();
        if (state === 'on') await configManager.set('highlight', true);
        else if (state === 'off') await configManager.set('highlight', false);
        else logWarning('Usage: /highlight on|off');
    },
    debug: async (args) => {
        const state = args.trim().toLowerCase();
        if (state === 'on') await configManager.set('debug', true);
        else if (state === 'off') await configManager.set('debug', false);
        else logWarning('Usage: /debug on|off');
    },
};

// Editor Interaction
async function openInEditor(content) {
    const tempFile = path.join(os.tmpdir(), `${TEMP_FILE_PREFIX}${Date.now()}.md`);
    try {
        logSystem(`Opening content in editor (${EDITOR}). Save and close editor when finished.`);
        await fs.writeFile(tempFile, content, 'utf8');
        if (readlineInterface) readlineInterface.pause();
        await new Promise((resolve, reject) => {
            const editorParts = EDITOR.split(' ');
            const editorCmd = editorParts[0];
            const editorArgs = [...editorParts.slice(1), tempFile];
            logDebug(`Spawning editor: ${editorCmd} ${editorArgs.join(' ')}`);
            const proc = spawn(editorCmd, editorArgs, { stdio: 'inherit', shell: process.platform === 'win32' });
            proc.on('error', reject);
            proc.on('close', (code) => {
                logDebug(`Editor exited with code: ${code}`);
                code === 0 ? resolve() : resolve(logWarning(`Editor exited with code ${code}. Reading file anyway.`));
            });
        });
        if (readlineInterface) readlineInterface.resume();
        const updated = await fs.readFile(tempFile, 'utf8');
        logSystem('Editor closed. Content read.');
        safePromptRefresh();
        return updated;
    } catch (error) {
        logError(`Editor interaction failed: ${error.message}`);
        safePromptRefresh();
        return null;
    } finally {
        try { await fs.unlink(tempFile); } catch (unlinkErr) { logWarning(`Failed to delete temp file: ${tempFile}`, unlinkErr.message); } // Added error detail for unlink fail - Pyrmethus
    }
}

// Configuration and Initialization
async function applyConfigChange(key) {
    if (['apiKey', 'model', 'temperature', 'systemPrompt'].includes(key)) {
        await initializeModelInstance();
    }
}

async function initializeModelInstance() {
    if (!API_KEY) {
        logError('API key is required. Set GOOGLE_API_KEY environment variable or use --api-key argument.'); // More informative API key error - Pyrmethus
        throw new Error('API key is required.'); // Ensure startup fails if no API key
    }
    try {
        genAI = new GoogleGenerativeAI(API_KEY);
        model = genAI.getGenerativeModel({
            model: MODEL_NAME,
            generationConfig: { temperature: TEMPERATURE },
            safetySettings: SAFETY_SETTINGS,
            systemInstruction: SYSTEM_PROMPT
        });
        chatSession = model.startChat({ history: HISTORY });
        logSystem(`Initialized ${MODEL_NAME} with temperature ${TEMPERATURE}`);
    } catch (e) {
        logError(`Failed to initialize AI model: ${e.message}`); // More informative AI init error - Pyrmethus
        throw e; // Re-throw to prevent startup
    }
}

async function gracefulExit(code = 0) {
    logSystem('Shutting down...');
    if (readlineInterface) readlineInterface.close();
    await saveHistory();
    await saveMacros();
    logMuted('State saved.'); // Muted log for state saved - Pyrmethus
    process.exit(code);
}

// Task Queue
async function queueTask(task, priority = false) {
    task.queuedAt = Date.now();
    if (priority) commandQueue.unshift(task);
    else commandQueue.push(task);
    logDebug(`Task queued: ${task.type} (priority: ${priority}). Queue length: ${commandQueue.length}`);
    if (!isProcessingQueue) process.nextTick(processQueue);
}

async function processQueue() {
    if (isProcessingQueue || !commandQueue.length) return;
    isProcessingQueue = true;
    const task = commandQueue.shift();
    logDebug(`Processing task: ${task.type}`);
    try {
        if (task.type === 'command') await task.handler(task.args);
        else if (task.type === 'message') await task.handler(task.parts);
    } catch (error) {
        logError(`Task failed: ${error.message}`);
    }
    isProcessingQueue = false;
    if (commandQueue.length) process.nextTick(processQueue);
}

// Input Handling
function parseCommand(line) {
    if (!line.startsWith(CMD_PREFIX)) return null;
    const match = line.match(/^\/([a-zA-Z]+)(?:\s+(.+))?$/s);
    if (!match) return null;
    return { command: match[1].toLowerCase(), args: match[2] || '' };
}

async function handleLineInput(line) {
    const finalizedLine = line.trim();
    if (!finalizedLine) return safePromptRefresh();
    if (isPastingMode) {
        pasteBuffer.push(finalizedLine);
        return safePromptRefresh();
    }
    const expanded = expandMacro(finalizedLine);
    const cmd = parseCommand(expanded || finalizedLine);
    if (cmd && commandHandlers.hasOwnProperty(cmd.command)) {
        const priorityCommands = ['exit', 'quit', 'bye'];
        const isPriority = priorityCommands.includes(cmd.command);
        await queueTask({ type: 'command', handler: commandHandlers[cmd.command], args: cmd.args, sourceLine: finalizedLine }, isPriority);
    } else if (finalizedLine) {
        await queueTask({ type: 'message', handler: sendMessageToAI, parts: [{ text: finalizedLine }], sourceLine: finalizedLine });
    }
}

// Main Function
async function main() {
    process.on('SIGINT', () => gracefulExit());
    process.on('uncaughtException', (err) => { logError('Uncaught exception:', err); gracefulExit(1); });
    process.on('unhandledRejection', (err) => { logError('Unhandled promise rejection:', err); gracefulExit(1); }); // Pyrmethus - Handle unhandled promise rejections

    await configManager.load();
    await loadHistory();
    await loadMacros();
    try { // Try-catch around model initialization to catch startup errors - Pyrmethus
        await initializeModelInstance();
    } catch (startupError) {
        logError('NeonCLI startup failed due to AI model initialization error.'); // More user-friendly startup error - Pyrmethus
        return; // Exit if model initialization fails
    }


    readlineInterface = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: '',
        completer: (line) => {
            const completions = [];
            const currentWord = line.split(/\s+/).pop() || '';
            if (line.startsWith(CMD_PREFIX)) {
                const cmdPart = currentWord.substring(1).toLowerCase();
                Object.keys(commandHandlers)
                    .filter(c => c.startsWith(cmdPart))
                    .forEach(c => completions.push(`/${c}`));
            } else if (line.startsWith(MACRO_PREFIX)) {
                const macroPart = currentWord.substring(1).toLowerCase();
                Object.keys(macros)
                    .filter(m => m.startsWith(macroPart))
                    .forEach(m => completions.push(`!${m}`));
            }
            const hits = completions.filter(c => c.startsWith(currentWord));
            return [hits.length ? hits : completions, currentWord];
        }
    });

    readlineInterface.on('line', handleLineInput);
    readlineInterface.on('close', () => gracefulExit());
    logSystem(`${APP_NAME} v${APP_VERSION} started. Type /help for commands.`);
    safePromptRefresh();
}

main().catch(err => { logError('Startup critical failure:', err); process.exit(1); }); // Catch any remaining startup errors - Pyrmethus
