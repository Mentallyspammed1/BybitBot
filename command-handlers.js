import chalk from 'chalk';
import path from 'path';
import {
    logSystem, logWarning, logError, neon, clearConsole, checkFileExists,
    ensureDirectoryExists, sendTermuxToast, estimateTokenCountLocal, openInEditor,
    executeShellCommand, executePythonCode, saveChatHistory, saveMacros, loadMacros,
    convertFileToGenerativePart, sendMessageToAI, isAiThinking, isWaitingForConfirmation,
    configManager, chatHistory, HISTORY_FILE, MACROS_FILE, CMD_PREFIX, MACRO_PREFIX,
    MODEL_NAME, SAFETY_MAP, SAFETY_MAP_VALUES, termuxToastAvailable, readlineInterface,
    isPastingMode, pasteBufferContent, lastUserTextInput, spinner, isWaitingForShellConfirmation,
    pendingShellCommand, isWaitingForPythonConfirmation, pendingPythonCode, macros,
    IS_SHELL_ALLOWED, IS_PYTHON_ALLOWED, IS_HIGHLIGHTING_ACTIVE, APP_VERSION, APP_NAME,
    DEFAULT_SYSTEM_PROMPT_TEXT, initializeModelInstance, gracefulExit, safePromptRefresh,
    estimateTokenCountAPI, MAX_HISTORY_PAIRS, trimHistory, KNOWN_MODELS, USE_SYSTEM_PROMPT,
    applyHighlightingPrint,
} from './index.js'; // Import from index.js (main file)
import fs from 'fs/promises'; // Import fs for file operations in handlers

// --- Command Handlers ---

// Help Command
export const help = () => { // ... (rest of help - same as v1.9.0)
    logSystem(`\n--- ${APP_NAME} v${APP_VERSION} Help ---`);
    const categories = {
         "Chat & History": [
             [`${CMD_PREFIX}edit`, "Edit the last user message and resubmit"],
             [`${CMD_PREFIX}regen`, "Regenerate the last AI response"],
             [`${CMD_PREFIX}paste`, "Start multi-line paste mode"],
             [`${CMD_PREFIX}endpaste`, "End multi-line paste mode and send"],
             [`${CMD_PREFIX}history [num]`, "Show chat history (last [num] turns, default 10)"],
             [`${CMD_PREFIX}search <query>`, "Search chat history for <query> (case-insensitive)"],
             [`${CMD_PREFIX}clear`, "Clear chat history (memory & file)"],
             [`${CMD_PREFIX}tokens`, "Estimate token count (local heuristic + API call)"],
             [`${CMD_PREFIX}context <num>`, `Set max history turns (current: ${MAX_HISTORY_PAIRS})`],
         ],
         "Input & Output": [
             [`${CMD_PREFIX}file, ${CMD_PREFIX}f, ${CMD_PREFIX}l <path> [prompt]`, "Load file content & optionally prompt AI"],
             [`${CMD_PREFIX}save <filename>`, "Save *next* AI response to file (overwrites)"],
         ],
         "Model & Generation": [
             [`${CMD_PREFIX}model [name]`, `Show/Switch model (current: ${MODEL_NAME})`],
             [`${CMD_PREFIX}model list`, `List available models`],
             [`${CMD_PREFIX}model reload`, `Reload model config (applies /system changes)`],
             [`${CMD_PREFIX}temp <value>`, `Set temperature (current: ${configManager.get('temperature')}, 0.0-2.0)`],
             [`${CMD_PREFIX}safety [level]`, `Display/Set safety threshold (e.g., /safety block_none)`],
             [`${CMD_PREFIX}system view`, "View the current system prompt in use"],
             [`${CMD_PREFIX}system edit`, "Edit system prompt in $EDITOR (requires /model reload)"],
             [`${CMD_PREFIX}system set <prompt...>`, "Set system prompt (saved, requires /model reload)"],
             [`${CMD_PREFIX}system reset`, "Reset system prompt to default (saved, requires /model reload)"],
             [`${CMD_PREFIX}system toggle`, `Toggle system prompt on/off (now: ${USE_SYSTEM_PROMPT ? 'ON' : 'OFF'})`],
         ],
         "Execution (Use With Caution!)": [
             [`${CMD_PREFIX}shell <on|off>`, `Toggle AI shell execution ability (now: ${IS_SHELL_ALLOWED ? chalk.green('ON') : chalk.red('OFF')})`],
             [`${CMD_PREFIX}python <on|off>`, `Toggle AI python execution ability (now: ${IS_PYTHON_ALLOWED ? chalk.green('ON') : chalk.red('OFF')})`],
             [`${CMD_PREFIX}shell run <command...>`, `Run a shell command directly (requires confirmation)`],
             [`${CMD_PREFIX}python run <code...>`, `Run python code directly (requires confirmation)`],
             [`${CMD_PREFIX}shell save <file> <command...>`, `Run shell command & save output to file`],
             [`${CMD_PREFIX}python save <file> <code>`, `Run python code & save output to file`],
             [`${CMD_PREFIX}env set <key> <value>`, `Set environment variable for shell/python`],
             [`${CMD_PREFIX}env unset <key>`, `Unset environment variable`],
         ],
         "Macros (!name to expand)": [
             [`${CMD_PREFIX}macro define <name> <text...>`, "Define a macro"],
             [`${CMD_PREFIX}macro undef <name>`, "Delete a macro"],
             [`${CMD_PREFIX}macro list`, "List defined macros"],
             [`${CMD_PREFIX}macro save`, "Save macros to file manually"],
             [`${CMD_PREFIX}macro load`, "Load/reload macros from file"],
         ],
         "Configuration & Meta": [
             [`${CMD_PREFIX}config list`, "List current configuration settings (excl. API key)"],
             [`${CMD_PREFIX}config set <key> <value>`, "Set a configuration value (saved to file)"],
             [`${CMD_PREFIX}config safety <level>`, `Set safety level (BLOCK_NONE, LOW_AND_ABOVE, etc.)`],
             [`${CMD_PREFIX}highlight <on|off>`, `Toggle syntax highlighting (now: ${IS_HIGHLIGHTING_ACTIVE ? 'ON' : 'OFF'})`],
             [`${CMD_PREFIX}debug <on|off>`, `Toggle debug logging (now: ${IS_DEBUG_MODE ? 'ON' : 'OFF'})`],
             [`${CMD_PREFIX}help, ${CMD_PREFIX}?`, "Display this help message"],
             [`${CMD_PREFIX}exit, ${CMD_PREFIX}quit, ${CMD_PREFIX}q`, "Exit the chat session"],
         ]
    };
    for (const [category, commands] of Object.entries(categories)) {
        console.log(`\n${neon.systemInfo(`--- ${category} ---`)}`);
        const pad = commands.reduce((max, line) => Math.max(max, line[0].length), 0) + 2;
        commands.forEach(([cmd, desc]) => console.log(`${neon.commandHelp(cmd.padEnd(pad))}${neon.systemInfo(desc)}`));
    }
    console.log(neon.systemInfo("\nTips:"));
    console.log(neon.systemInfo(` - Use ${chalk.cyanBright('Up/Down')} arrows to navigate command history.`));
    console.log(neon.systemInfo(` - Use ${chalk.cyanBright('Tab')} for command/macro/config key completion.`));
    console.log(neon.systemInfo(` - Use ${MACRO_PREFIX}name [args] to expand a macro with optional arguments appended.`));
    if (!process.argv.includes('--allow-shell')) logSystem(neon.warning(`\nNote: Enabling shell execution requires the --allow-shell startup flag.`));
    if (!process.argv.includes('--allow-python')) logSystem(neon.warning(`Note: Enabling Python execution requires the --allow-python startup flag.`));
    console.log(neon.systemInfo("---------------------------------\n"));
};

// Exit Command
export const exit = async () => { // ... (rest of exit - same as v1.9.0)
    await gracefulExit();
};

// Clear Command
export const clear = async () => { // ... (rest of clear - same as v1.9.0)
    if (isAiThinking || isWaitingForConfirmation()) {
        logWarning("Cannot clear history while AI is busy or waiting for confirmation.");
        return;
    }
    chatHistory = [];
    if (aiModelInstance) {
        await initializeModelInstance(false, true);
        logSystem('Chat history cleared from memory and session restarted.');
    } else {
        logWarning('AI Model not initialized, history cleared in memory only.');
    }

    if (HISTORY_FILE && await checkFileExists(HISTORY_FILE)) {
        try {
            await fs.unlink(HISTORY_FILE);
            logSystem(`History file ${neon.filePath(HISTORY_FILE)} deleted.`);
        } catch (error) {
            logError(`Could not delete history file ${neon.filePath(HISTORY_FILE)}`, error);
        }
    }
};

// History Command
export const history = (args) => { // ... (rest of history - same as v1.9.0)
    const numTurnsToShow = parseInt(args, 10) || 10;
    if (isNaN(numTurnsToShow) || numTurnsToShow < 1) {
        logWarning("Invalid number of turns. Please provide a positive integer.");
        return;
    }
    logSystem("\n--- Chat History ---");
    if (chatHistory.length === 0) {
        console.log(neon.warning('(Empty history)'));
        logSystem("--------------------\n");
        return;
    }

    const numEntriesToShow = Math.min(numTurnsToShow * 2, chatHistory.length);
    const startIndex = Math.max(0, chatHistory.length - numEntriesToShow);

    for (let i = startIndex; i < chatHistory.length; i++) {
        const message = chatHistory[i];
        if (!message || !message.role || !message.parts) continue;

        const turnNumber = Math.floor(i / 2) + 1;
        const roleMarker = message.role === ROLE_USER ? neon.promptMarker : neon.aiMarker;
        const contentPreview = message.parts
            .map(part => part.text || (part.inlineData ? `[${part.inlineData.mimeType || 'Inline Data'}]` : '[Non-Text Part]'))
            .join('')
            .replace(/\n+/g, ' ')
            .slice(0, 150);
        const totalLength = message.parts?.reduce((len, p) => len + (p.text?.length || 0), 0) || 0;
        const ellipsis = totalLength > 150 ? '...' : '';
        const colorizer = message.role === ROLE_USER ? neon.userPrompt : neon.aiResponse;

        console.log(`${roleMarker}${colorizer(`(Turn ${turnNumber}): ${contentPreview}${ellipsis}`)}`);
    }

    if (startIndex > 0) {
        console.log(neon.systemInfo(`... (showing last ${Math.ceil(numEntriesToShow / 2)} of ${Math.ceil(chatHistory.length / 2)} total turns)`));
    }
    logSystem("--------------------\n");
};

// File Command
export const file = async (args) => { // ... (rest of file - same as v1.9.0)
    if (!args) {
        logWarning(`Usage: ${CMD_PREFIX}file <file_path> [optional prompt text]`);
        return;
    }
    const firstSpaceIndex = args.indexOf(' ');
    const filePath = (firstSpaceIndex === -1 ? args : args.substring(0, firstSpaceIndex)).trim();
    const userPrompt = (firstSpaceIndex === -1 ? '' : args.substring(firstSpaceIndex + 1).trim());

    if (!filePath) {
         logWarning(`Usage: ${CMD_PREFIX}file <file_path> [optional prompt text]`);
         return;
     }

    const filePart = await convertFileToGenerativePart(filePath);

    if (filePart) {
        const messageParts = [filePart];
        const promptText = userPrompt || `The user provided this file (${path.basename(filePath)}). Please describe or analyze it as appropriate.`;
        messageParts.push({ text: promptText });

        logSystem(`Sending file ${neon.filePath(path.basename(filePath))} with prompt...`);
        await sendMessageToAI(messageParts);
    }
};

// Paste Commands
export const paste = () => { // ... (rest of paste - same as v1.9.0)
    if (isPastingMode) { logWarning("Already in paste mode. Type /endpaste to finish."); return; }
    isPastingMode = true;
    pasteBufferContent = [];
    logSystem('Paste mode activated. Enter text line by line. Type /endpaste on a new line to send.');
    readlineInterface.setPrompt(neon.pasteMarker);
    readlineInterface.prompt(true);
};
export const endpaste = async () => { // ... (rest of endpaste - same as v1.9.0)
    if (!isPastingMode) { logWarning(`Not in paste mode. Use ${CMD_PREFIX}paste first.`); return; }
    isPastingMode = false;
    readlineInterface.setPrompt(neon.promptMarker);
    const content = pasteBufferContent.join('\n');
    pasteBufferContent = [];
    if (content.trim()) {
        logSystem('Sending pasted content...');
        await sendMessageToAI([{ text: content }]);
    } else {
        logSystem('Paste mode ended. No content sent.');
        safePromptRefresh();
    }
};

// Edit Command
export const edit = async () => { // ... (rest of edit - same as v1.9.0)
    if (isAiThinking || isWaitingForConfirmation()) { logWarning("Cannot edit while AI is busy or waiting for confirmation."); return; }
    if (chatHistory.length < 1) { logWarning("No history to edit."); return; }

    let lastUserIndex = -1;
    for (let i = chatHistory.length - 1; i >= 0; i--) {
        if (chatHistory[i]?.role === ROLE_USER) {
            lastUserIndex = i;
            break;
        }
    }

    if (lastUserIndex === -1) { logWarning("Could not find a previous user message in history to edit."); return; }

    const lastUserEntry = chatHistory[lastUserIndex];
    const lastUserText = lastUserEntry.parts?.filter(p => p.text).map(p => p.text).join('\n') || '';

    if (!lastUserText && lastUserEntry.parts?.some(p => p.inlineData)) {
         logWarning("Last user message contained file/inline data which cannot be edited directly. Send a new message instead.");
         return;
     } else if (!lastUserText) {
         logWarning("Last user message had no text content to edit.");
         return;
     }

    logSystem("Removing last exchange and preparing for edit...");
    let itemsToRemove = chatHistory.length - lastUserIndex;
    chatHistory.splice(lastUserIndex, itemsToRemove);

    await saveChatHistory();

     logSystem(`Previous message loaded. Edit and press Enter.`);
     if (readlineInterface) {
         readlineInterface.write(null, { ctrl: true, name: 'u' });
         readlineInterface.write(lastUserText);
     }
};


// Regen Command
export const regen = async () => { // ... (rest of regen - same as v1.9.0)
    if (isAiThinking || isWaitingForConfirmation()) { logWarning("Cannot regenerate while AI is busy or waiting for confirmation."); return; }
    if (chatHistory.length < 1) { logWarning("No previous user message in history to regenerate response for."); return; }

     let lastUserIndex = -1;
     for (let i = chatHistory.length - 1; i >= 0; i--) {
         if (chatHistory[i]?.role === ROLE_USER) {
             lastUserIndex = i;
             break;
         }
     }

    if (lastUserIndex === -1) { logWarning("Could not find a previous user message in history."); return; }

    const lastUserEntry = chatHistory[lastUserIndex];

    let itemsToRemove = chatHistory.length - 1 - lastUserIndex;
    if (itemsToRemove < 0) itemsToRemove = 0;

    logSystem("Removing last AI response(s) and regenerating...");
    if (itemsToRemove > 0) {
        chatHistory.splice(lastUserIndex + 1, itemsToRemove);
        await saveChatHistory();
    } else {
        logWarning("No AI response found after the last user message to remove.");
    }

    await sendMessageToAI(lastUserEntry.parts, false, true);
};


// Save Command
export const save = async (args) => { // ... (rest of save - same as v1.9.0)
    if (!args) { logWarning(`Usage: ${CMD_PREFIX}save <filename>`); return; }
    const potentialPath = path.resolve(args.trim());

    try {
        await ensureDirectoryExists(potentialPath);

        if (await checkFileExists(potentialPath)) {
            logWarning(`File ${neon.filePath(potentialPath)} exists and will be overwritten.`);
        }

        saveFilePath = potentialPath;
        logSystem(`Next AI response will be saved to ${neon.filePath(saveFilePath)}`);
    } catch (error) {
        logError(`Cannot set save path to ${neon.filePath(potentialPath)}. Check permissions or path validity.`, error);
        saveFilePath = null;
    }
};

// Temperature Command
export const temp = async (args) => { // ... (rest of temp - same as v1.9.0)
    const temperature = parseFloat(args);
    if (isNaN(temperature) || temperature < 0.0 || temperature > 2.0) { // Allow 2.0 now
        logWarning(`Invalid temperature value. Must be between 0.0 and 2.0. (Current: ${configManager.get('temperature')})`);
        return;
    }
    await configManager.set('temperature', args);
};

// Model Command
export const model = async (args) => { // ... (rest of model - same as v1.9.0)
    const actionOrModel = args?.trim().toLowerCase();

    if (actionOrModel === 'reload') {
        logSystem("Reloading model configuration (applying system prompt, safety, etc)...");
        await initializeModelInstance(false, true);
        return;
    } else if (actionOrModel === 'list') {
         logSystem("\n--- Available Models ---");
         KNOWN_MODELS.forEach(modelName => {
             const isDefault = modelName === configManager.get('modelName');
             console.log(`${neon.filePath(modelName)}${isDefault ? chalk.gray(' (default)') : ''}`);
         });
         logSystem("----------------------\n");
        return;
    }

    const currentModel = configManager.get('modelName');
    if (!actionOrModel) {
        logSystem(`Current model: ${neon.filePath(currentModel)}`);
        logSystem(`Use ${CMD_PREFIX}model <name> to switch, or ${CMD_PREFIX}model reload to apply config.`);
        logSystem(`Use ${CMD_PREFIX}model list to see available models.`);
        return;
    }

    if (actionOrModel === currentModel) {
        logSystem(`Already using model: ${neon.filePath(currentModel)}`);
        return;
    }

    await configManager.set('modelName', actionOrModel);
};

// Safety Command (Updated)
export const safety = async (args) => { // ... (rest of safety - same as v1.9.0)
    const level = args?.trim().toUpperCase();

    if (!level) {
        const currentSafety = configManager.get('safety');
        logSystem(`Current safety threshold: ${neon.filePath(currentSafety)} (${SAFETY_MAP[currentSafety] || 'Unknown Threshold'})`);
        logSystem(`Blocks content rated >= ${currentSafety.replace('BLOCK_', '').replace('_AND_ABOVE', '')}`);
        logSystem(`Categories: HARM_CATEGORY_HARASSMENT, HATE_SPEECH, SEXUALLY_EXPLICIT, DANGEROUS_CONTENT`);
        logSystem(`\nUsage: ${CMD_PREFIX}safety <level>`);
        logSystem(`Levels: ${Object.keys(SAFETY_MAP).join(', ')}`);
        logSystem(`Example: ${CMD_PREFIX}safety block_medium_and_above`);
        logSystem(`Requires '/model reload' to apply changes.`);
        return;
    }

    if (!SAFETY_MAP[level]) {
        logWarning(`Invalid safety level: "${level}". Allowed levels: ${Object.keys(SAFETY_MAP).join(', ')}`);
        return;
    }

    await configManager.set('safety', level.toUpperCase());
};

// Debug Command
export const debug = async (args) => { // ... (rest of debug - same as v1.9.0)
    const currentVal = configManager.get('debug');
    const newVal = args ? args.toLowerCase() : (currentVal ? 'off' : 'on');
    await configManager.set('debug', newVal);
};

// Highlight Command
export const highlight = async (args) => { // ... (rest of highlight - same as v1.9.0)
     const currentVal = configManager.get('highlight');
     const newVal = args ? args.toLowerCase() : (currentVal ? 'off' : 'on');
    await configManager.set('highlight', newVal);
};

// Search Command
export const search = (query) => { // ... (rest of search - same as v1.9.0)
    if (!query) { logWarning(`Usage: ${CMD_PREFIX}search <query>`); return; }
    const lowerQuery = query.toLowerCase();
    const queryRegex = new RegExp(query.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'), 'gi');
    let matches = [];

    chatHistory.forEach((entry, index) => {
        if (entry.parts?.some(p => p.text?.toLowerCase().includes(lowerQuery))) {
            matches.push({ entry, index });
        }
    });

    logSystem(`\n--- Search Results for "${query}" (${matches.length} matches) ---`);
    if (matches.length === 0) {
        console.log(neon.warning('(No matches found)'));
    } else {
        matches.forEach(({ entry, index }) => {
            const turnNumber = Math.floor(index / 2) + 1;
            const roleMarker = entry.role === ROLE_USER ? neon.promptMarker : neon.aiMarker;
            const fullText = entry.parts?.map(p => p.text || '').join('') || '[Non-text or empty]';
            const previewText = fullText.replace(/\n+/g, ' ').slice(0, 200);
            const ellipsis = fullText.length > 200 ? '...' : '';
            const highlightedPreview = previewText.replace(queryRegex, (match) => neon.searchHighlight(match));
            const colorizer = entry.role === ROLE_USER ? neon.userPrompt : neon.aiResponse;

            console.log(`${roleMarker}${colorizer(`(Turn ${turnNumber}): ${highlightedPreview}${ellipsis}`)}`);
        });
    }
    logSystem("-------------------------------------------\n");
};

// Shell Execution Toggle/Run Command
export const shell = async (args) => { // ... (rest of shell - same as v1.9.0)
     const parts = args.trim().split(' ');
     const action = parts[0]?.toLowerCase();
     const commandToRun = parts.slice(1).join(' ');

     if (action === 'on' || action === 'off') {
         const isEnabling = (action === 'on');
         if (!argv.allowShell && isEnabling) {
             logError(`Cannot enable shell execution via command without the --allow-shell startup flag.`);
             return;
         }
         await configManager.set('allowShell', action);
     } else if (action === 'run') {
         if (!IS_SHELL_ALLOWED) { logError("Shell execution is disabled. Use '/shell on' first (if --allow-shell flag was used)."); return; }
         if (!commandToRun) { logWarning("Usage: /shell run <command...>"); return; }
         if (isAiThinking || isWaitingForConfirmation()) { logWarning("Cannot execute while AI is busy or waiting for confirmation."); return; }

         pendingShellCommand = commandToRun;
         isWaitingForShellConfirmation = true;
         if (await confirmExecution('Shell Command (User Input)', pendingShellCommand)) {
             try {
                 const { stdout, stderr } = await executeShellCommand(pendingShellCommand);
                 let feedback = `${neon.shellMarker}Direct shell execution finished.`;
                 if (stdout) feedback += `\n${neon.sysMarker}Stdout:\n${neon.shellOutput(stdout)}`;
                 if (stderr) feedback += `\n${neon.warnMarker}Stderr:\n${neon.shellOutput(stderr)}`;
                 logSystem(feedback);
             } catch (error) { /* Error already logged by executeShellCommand */ }
         } else {
             logSystem(`${neon.shellMarker}Shell execution cancelled by user.`);
         }
         pendingShellCommand = null;
         isWaitingForShellConfirmation = false;
         safePromptRefresh();
     } else {
         const currentVal = configManager.get('allowShell');
         logSystem(`Shell execution ability: ${currentVal ? neon.commandHelp('enabled') : neon.warning('disabled')}. Use ${CMD_PREFIX}shell on|off to toggle.`);
     }
 };


// Python Execution Toggle/Run Command
export const python = async (args) => { // ... (rest of python - same as v1.9.0)
     const parts = args.trim().split(' ');
     const action = parts[0]?.toLowerCase();
     const codeToRun = parts.slice(1).join(' ');

     if (action === 'on' || action === 'off') {
         const isEnabling = (action === 'on');
         if (!argv.allowPython && isEnabling) {
             logError(`Cannot enable python execution via command without the --allow-python startup flag.`);
             return;
         }
         await configManager.set('allowPython', action);
     } else if (action === 'run') {
         if (!IS_PYTHON_ALLOWED) { logError("Python execution is disabled. Use '/python on' first (if --allow-python flag was used)."); return; }
         if (!codeToRun) { logWarning("Usage: /python run <code>"); return; }
         if (isAiThinking || isWaitingForConfirmation()) { logWarning("Cannot execute while AI is busy or waiting for confirmation."); return; }

         pendingPythonCode = codeToRun;
         isWaitingForPythonConfirmation = true;
         if (await confirmExecution('Python Code (User Input)', pendingPythonCode)) {
             try {
                 const { stdout, stderr } = await executePythonCode(pendingPythonCode);
                 let feedback = `${neon.pythonMarker}Direct python execution finished.`;
                 if (stdout) feedback += `\n${neon.sysMarker}Stdout:\n${neon.pythonOutput(stdout)}`;
                 if (stderr) feedback += `\n${neon.warnMarker}Stderr:\n${neon.pythonOutput(stderr)}`;
                 logSystem(feedback);
             } catch (error) { /* Error already logged by executePythonCode */ }
         } else {
             logSystem(`${neon.pythonMarker}Python execution cancelled by user.`);
         }
         pendingPythonCode = null;
         isWaitingForPythonConfirmation = false;
         safePromptRefresh();
     } else {
         const currentVal = configManager.get('allowPython');
         logSystem(`Python execution ability: ${currentVal ? neon.commandHelp('enabled') : neon.warning('disabled')}. Use ${CMD_PREFIX}python <on|off> to toggle.`);
     }
 };

// Macro Command
export const macro = async (args) => { // ... (rest of macro - same as v1.9.0)
    const parts = args.trim().split(' ');
    const action = parts[0]?.toLowerCase();
    const name = parts[1];
    const content = parts.slice(2).join(' ');

    switch (action) {
        case 'define':
        case 'set':
            if (!name || !content) { logWarning(`Usage: ${CMD_PREFIX}macro define <name> <text...>`); return; }
            if (!VALID_MACRO_NAME_REGEX.test(name)) { logWarning(`Invalid macro name "${name}". Use only letters, numbers, underscores, hyphens.`); return; }
            if (commandHandlers[name] || ['help', '?'].includes(name)) { logWarning(`Macro name "${name}" conflicts with a built-in command.`); return; }

            if (macros[name]) logWarning(`Overwriting macro !${neon.macroName(name)}.`);
            else logSystem(`Defining macro !${neon.macroName(name)}.`);
            macros[name] = content;
            await saveMacros(); break;
        case 'undef':
        case 'delete':
        case 'remove':
        case 'del':
            if (!name) { logWarning(`Usage: ${CMD_PREFIX}macro undef <name>`); return; }
            if (macros[name]) {
                delete macros[name];
                logSystem(`Macro !${neon.macroName(name)} deleted.`);
                await saveMacros();
            } else {
                logWarning(`Macro !${neon.macroName(name)} not found.`);
            }
            break;
        case 'list':
        case 'ls':
            logSystem("\n--- Defined Macros ---");
            const macroNames = Object.keys(macros);
            if (macroNames.length === 0) {
                console.log(neon.warning('(No macros defined)'));
            } else {
                macroNames.sort().forEach(mName => {
                    const preview = macros[mName].length > 60 ? macros[mName].slice(0, 57) + '...' : macros[mName];
                    console.log(`${MACRO_PREFIX}${neon.macroName(mName.padEnd(15))} ${neon.macroContent(preview)}`);
                });
            }
            logSystem("--------------------\n");
            break;
        case 'save':
            await saveMacros();
            logSystem(`Macros saved to ${neon.filePath(MACROS_FILE)}.`);
            break;
        case 'load':
            await loadMacros();
            break;
        default:
            logWarning(`Unknown macro action: ${action}. Use define, undef, list, save, load.`);
    }
};

// Config Command
export const config = async (args) => { // ... (rest of config - same as v1.9.0)
    const parts = args.trim().split(' ');
    const action = parts[0]?.toLowerCase();
    const key = parts[1];
    const value = parts.slice(2).join(' ');

    if (action === 'list' || !action) {
        logSystem("\n--- Current Configuration ---");
        const currentConfig = configManager.getAll();
        const defaults = configManager.getDefaults();
        const longestKey = Object.keys(currentConfig).reduce((max, k) => Math.max(max, k.length), 0);

        Object.keys(defaults)
            .filter(k => k !== 'apiKey')
            .sort()
            .forEach(k => {
                const currentValue = currentConfig[k];
                const isDefault = currentValue === defaults[k];
                const valueStr = JSON.stringify(currentValue);
                console.log(`${neon.configKey(k.padEnd(longestKey + 2))}` +
                            `${neon.configValue(valueStr)}` +
                            `${isDefault ? chalk.gray(' (default)') : ''}`);
            });
        logSystem("---------------------------\n");
    } else if (action === 'set') {
        if (!key || value === undefined) {
            logWarning(`Usage: ${CMD_PREFIX}config set <key> <value>`);
            return;
        }
        await configManager.set(key, value);
    }  else if (action === 'safety') {
         const safetyLevel = key?.toUpperCase();
         if (!safetyLevel) {
             return safety();
         }
         if (!SAFETY_MAP[safetyLevel]) {
             logWarning(`Invalid safety level: "${safetyLevel}". Allowed levels: ${Object.keys(SAFETY_MAP).join(', ')}`);
             return;
         }
         await configManager.set('safety', safetyLevel);
     } else {
        logWarning(`Unknown config action: ${action}. Use 'list' or 'set'.`);
    }
};

// System Prompt Command
export const system = async (args) => { // ... (rest of system - same as v1.9.0)
    const parts = args.trim().split(' ');
    const action = parts[0]?.toLowerCase();
    const promptText = parts.slice(1).join(' ');

    switch (action) {
        case 'view':
            logSystem("--- Current System Prompt ---");
            console.log(chalk.greenBright(configManager.get('systemPrompt')));
            logSystem("-----------------------------");
            break;
        case 'set':
            if (!promptText) { logWarning(`Usage: ${CMD_PREFIX}system set <prompt text...>`); return; }
            await configManager.set('systemPrompt', promptText);
            logWarning("System prompt updated in config. Use '/model reload' to apply it to the current session.");
            break;
        case 'edit':
            logSystem(`Opening current system prompt in default editor ($EDITOR)...`);
            const currentPrompt = configManager.get('systemPrompt');
            const editedPrompt = await openInEditor(currentPrompt);
            if (editedPrompt !== null && editedPrompt !== currentPrompt) {
                logSystem("Editor closed. Saving updated prompt...");
                await configManager.set('systemPrompt', editedPrompt);
                logWarning("System prompt updated in config. Use '/model reload' to apply it to the current session.");
            } else if (editedPrompt === currentPrompt) {
                logSystem("Editor closed. No changes detected.");
            } else {
                logError("Failed to get updated prompt from editor.");
            }
            break;
        case 'reset':
            await configManager.set('systemPrompt', DEFAULT_SYSTEM_PROMPT_TEXT);
            logWarning("System prompt reset to default in config. Use '/model reload' to apply it to the current session.");
            break;
        case 'toggle':
            const currentVal = configManager.get('useSystemPrompt');
            await configManager.set('useSystemPrompt', !currentVal);
            break;
        default:
            logWarning(`Unknown system prompt action: ${action}. Use view, set, edit, or reset.`);
    }
};

// Tokens Command (Local Estimate + API Call)
export const tokens = async () => { // ... (rest of tokens - same as v1.9.0)
    if (!aiModelInstance) {
        logError("AI Model not initialized. Cannot estimate tokens.");
        return;
    }
    if (isAiThinking || isWaitingForConfirmation()) {
        logWarning("Cannot estimate tokens while AI is busy or waiting for confirmation.");
        return;
    }

    let localEstimate = 0;
    chatHistory.forEach(entry => {
        entry.parts.forEach(part => {
            if (part.text) {
                localEstimate += estimateTokenCountLocal(part.text);
            }
        });
    });
    localEstimate += estimateTokenCountLocal(configManager.get('systemPrompt'));

    logSystem(`Local token estimate (heuristic): ~${neon.tokenCount(localEstimate)} tokens.`);
    logSystem(`Requesting accurate count from API...`);

    isAiThinking = true;
    spinner = ora({ text: 'Counting tokens...', color: neon.spinnerColor, stream: process.stdout }).start();

    try {
        const contentToCount = [
             ...chatHistory
        ];

        if (contentToCount.length === 0) {
             spinner.stop();
             logSystem("History is empty, API token count is 0.");
             isAiThinking = false;
             safePromptRefresh();
             return;
         }

        const { totalTokens } = await aiModelInstance.countTokens({ contents: contentToCount });
        spinner.succeed(`API token count for current history: ${neon.tokenCount(totalTokens)} tokens.`);

    } catch (error) {
        spinner.fail("API token count failed.");
        logError("Error counting tokens via API.", error);
    } finally {
        isAiThinking = false;
        safePromptRefresh();
    }
};

// Context Command - NEW
export const context = async (args) => { // ... (rest of context - same as v1.9.0)
    const numTurns = parseInt(args, 10);
    if (isNaN(numTurns) || numTurns < 1) {
        logWarning("Invalid number of turns. Must be a positive integer.");
        return;
    }
    await configManager.set('maxHistory', numTurns);
    logSystem(`Max history turns set to ${neon.configValue(numTurns)}.`);
    trimHistory();
};

// Environment Variable Command - NEW
export const env = async (args) => { // ... (rest of env - same as v1.9.0)
    const parts = args.trim().split(' ');
    const action = parts[0]?.toLowerCase();
    const key = parts[1];
    const value = parts.slice(2).join(' ');

    if (action === 'set') {
        if (!key || !value) {
            logWarning(`Usage: ${CMD_PREFIX}env set <key> <value>`);
            return;
        }
        process.env[key] = value;
        logSystem(`Environment variable ${neon.configKey(key)} set.`);
         logSystem(neon.warning(`Note: Environment variables set via /env are only effective for shell and python commands *run from within NeonCLI*. They do not affect the parent shell environment.`));

    } else if (action === 'unset') {
        if (!key) {
            logWarning(`Usage: ${CMD_PREFIX}env unset <key>`);
            return;
        }
        delete process.env[key];
        logSystem(`Environment variable ${neon.configKey(key)} unset.`);
    } else {
        logWarning(`Usage: ${CMD_PREFIX}env set <key> <value> or ${CMD_PREFIX}env unset <key>`);
    }
};
