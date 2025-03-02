#!/bin/bash

# Create the templates directory if it doesn't exist
mkdir -p templates

# Create the files if they don't exist
touch templates/index.html templates/styles.css templates/script.js

# --- Writing to templates/index.html ---
echo $'<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> <!-- For responsive design -->
    <title>Enhanced Trading Bot Control</title>
    <link rel="stylesheet" href="styles.css"> <!-- External CSS -->
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Enhanced Trading Bot Control</h1>

        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="alert alert-info text-center" role="alert">
                    Bot Status: <span id="bot-status" class="mb-3">Awaiting Commands</span>
                </div>
            </div>
        </div>

        <div class="row justify-content-center mb-3">
            <div class="col-md-6">
                <div class="form-group">
                    <label for="tradingSymbol" class="form-label">Trading Symbol:</label>
                    <input type="text" class="form-control" id="tradingSymbol" placeholder="e.g., BTC/USDT" value="TRUMP/USDT">
                    <div id="symbol-validation-message" class="invalid-feedback"></div> <!-- Validation message area -->
                </div>
            </div>
        </div>

        <div class="row justify-content-center mb-4">
            <div class="col-md-8 d-grid gap-2 d-md-block">
                <div class="btn-group">
                    <button id="start-button" class="btn btn-neon" type="button" disabled> <!-- Start Disabled Initially -->
                        <i class="far fa-caret-up"></i>
                        Start Bot
                    </button>
                    <button id="stop-button" class="btn btn-neon" type="button" disabled> <!-- Stop Disabled Initially -->
                        <i class="fas fa-stop"></i>
                        Stop Bot
                    </button>
                    <a href="/logs" class="btn btn-neon" type="button" disabled>View Logs</a> <!-- Logs Disabled Initially -->
                </div>
                <div id="terminal-container" class="terminal-container">
                    <div id="terminal-output-area" aria-live="polite"> <!-- Accessibility: Announce terminal updates -->
                        <div class="terminal-line"><span class="terminal-prompt">Bot:</span> <span class="terminal-output">Awaiting commands...</span></div>
                    </div>
                </div>
            </div>
        </div>

        <div id="message" class="mt-3 text-center" role="alert" aria-live="assertive"></div> <!-- Message area for alerts - Accessibility -->
        <div id="progress-bar" class="progress mt-3 d-none"> <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%">Loading...</div></div>

    </div>

    <script src="script.js"></script> <!-- External JavaScript -->
</body>
</html>' > templates/index.html

# --- Writing to templates/styles.css ---
echo $'/* styles.css - Enhanced Styling for Trading Bot Control Panel */

/* Root variables for consistent theming */
:root {
    --dark-bg: #050505;
    --neon-green: #0f0;
    --neon-cyan: #00ffff;
    --neon-red: #f00;
    --terminal-font: \'Courier New\', monospace;
    --base-font-size: 1rem; /* Base font size for responsiveness */
}

body {
    padding-top: 20px;
    background-color: var(--dark-bg);
    color: #eee;
    font-family: var(--terminal-font);
    font-size: var(--base-font-size); /* Use root font size */
    margin: 0; /* Reset default margin */
}

.container {
    max-width: 960px; /* Adjust as needed */
    margin: 0 auto; /* Center container */
    padding: 20px;
}

.text-center {
    text-align: center;
}

.mb-3 {
    margin-bottom: 1rem;
}

.mb-4 {
    margin-bottom: 1.5rem;
}

.mt-3 {
    margin-top: 1rem;
}

.row {
    display: flex;
    flex-wrap: wrap; /* Allow items to wrap on smaller screens */
    justify-content: center; /* Center content in rows */
}

.justify-content-center {
    justify-content: center;
}

.col-md-8 {
    flex-basis: 100%; /* Full width on smaller screens */
    max-width: 100%; /* Full width on smaller screens */
}

@media (min-width: 768px) { /* Medium screens and up */
    .col-md-8 {
        flex-basis: 66.66666667%; /* Two-thirds width on medium screens and up */
        max-width: 66.66666667%;
    }
}

.col-md-6 {
    flex-basis: 100%; /* Full width on smaller screens */
    max-width: 100%; /* Full width on smaller screens */
}

@media (min-width: 768px) { /* Medium screens and up */
    .col-md-6 {
        flex-basis: 50%; /* Half width on medium screens and up */
        max-width: 50%;
    }
}

.d-grid {
    display: grid;
}

.gap-2 {
    gap: 0.5rem;
}

.d-md-block {
    display: block; /* For button group on larger screens */
}

@media (min-width: 768px) { /* Medium screens and up */
    .d-md-block {
        display: flex; /* Buttons in a row on medium screens and up */
        justify-content: center; /* Center buttons */
    }
}

.btn-group {
    display: flex;
    gap: 0.5rem; /* Spacing between buttons */
    justify-content: center; /* Center buttons */
    flex-wrap: wrap; /* Allow buttons to wrap on smaller screens */
}

.btn {
    padding: 0.5rem 1rem;
    border: 2px solid transparent; /* Default transparent border */
    border-radius: 7px;
    cursor: pointer;
    font-family: inherit; /* Inherit font from body */
    font-size: inherit; /* Inherit font size from body */
    transition: background-color 0.3s, color 0.3s, box-shadow 0.3s, border-color 0.3s; /* Smooth transitions */
    text-decoration: none; /* Remove default link underlines */
    display: inline-block; /* Ensure consistent button behavior */
    text-align: center; /* Center text in buttons */
}

.btn-neon {
    background-color: transparent;
    color: var(--neon-green);
    border-color: var(--neon-green);
    box-shadow: 0 0 15px var(--neon-green);
}

.btn-neon:hover, .btn-neon:focus {
    background-color: var(--neon-cyan);
    color: #000;
    box-shadow: 0 0 25px var(--neon-cyan);
    border-color: var(--neon-cyan);
}

.btn-neon:disabled {
    opacity: 0.5; /* Indicate disabled state */
    cursor: not-allowed; /* Indicate not clickable */
    box-shadow: none; /* Remove glow when disabled */
    background-color: transparent; /* Ensure no hover effect */
    color: var(--neon-green); /* Keep original color dimmed */
    border-color: var(--neon-green);
}

.alert {
    padding: 1rem;
    border-radius: 7px;
    margin-bottom: 1rem;
    border: 1px solid transparent; /* Default transparent border */
}

.alert-info {
    background-color: rgba(0, 123, 255, 0.1); /* Light blue background */
    color: #007bff; /* Blue text */
    border-color: #007bff;
}

.alert-success {
    background-color: rgba(40, 167, 69, 0.1); /* Light green background */
    color: #28a745; /* Green text */
    border-color: #28a745;
}

.alert-warning {
    background-color: rgba(255, 193, 7, 0.1); /* Light yellow background */
    color: #ffc107; /* Yellow text */
    border-color: #ffc107;
}

.alert-danger {
    background-color: rgba(220, 53, 69, 0.1); /* Light red background */
    color: #dc3545; /* Red text */
    border-color: #dc3545;
}

.form-group {
    margin-bottom: 1.5rem;
}

.form-label {
    display: block; /* Make label take full width */
    margin-bottom: 0.5rem;
    color: #eee; /* Ensure label is visible */
}

.form-control {
    display: block;
    width: 100%;
    padding: 0.5rem 0.75rem;
    font-size: inherit; /* Inherit font size */
    font-family: inherit; /* Inherit font family */
    line-height: 1.5;
    color: #eee;
    background-color: #222; /* Darker input background */
    border: 1px solid #444; /* Darker border */
    border-radius: 7px;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out; /* Smooth transition for focus */
}

.form-control:focus {
    color: #eee;
    background-color: #222;
    border-color: var(--neon-cyan); /* Neon cyan border on focus */
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(0, 207, 255, 0.25); /* Neon cyan focus ring */
}

.invalid-feedback {
    display: none; /* Hidden by default */
    width: 100%;
    margin-top: 0.25rem;
    font-size: 0.875rem;
    color: var(--neon-red); /* Neon red for error messages */
}

.form-control.is-invalid + .invalid-feedback {
    display: block; /* Show validation message when input is invalid */
}


/* Terminal Styling */
.terminal-container {
    margin-top: 20px;
    border: 2px solid var(--neon-green);
    padding: 15px;
    border-radius: 7px;
    background-color: #000;
    color: var(--neon-green);
    font-family: var(--terminal-font);
    font-size: 1em;
    overflow-y: scroll;
    height: 400px;
    box-shadow: 0 0 20px var(--neon-green);
    white-space: pre-wrap;
}

.terminal-line {
    margin-bottom: 6px;
}

.terminal-prompt {
    color: #00ff00;
    text-shadow: 0 0 5px #00ff00;
}

.terminal-output {
    color: var(--neon-green);
    text-shadow: 0 0 3px var(--neon-green);
}

.terminal-error {
    color: var(--neon-red);
    text-shadow: 0 0 8px var(--neon-red);
}

/* Progress Bar Styling */
.progress {
    background-color: #333; /* Dark progress bar background */
    border-radius: 7px;
    overflow: hidden; /* Ensure rounded corners for progress bar */
    height: 1rem; /* Adjust height as needed */
}

.progress-bar {
    background-color: var(--neon-cyan); /* Neon cyan progress bar color */
    color: #000; /* Black text on progress bar */
    text-align: center; /* Center progress text */
    white-space: nowrap; /* Prevent text wrapping */
    overflow: hidden; /* Hide text overflow */
    border-radius: 0; /* Remove border radius for progress bar itself */
}

.d-none {
    display: none !important; /* Utility class to hide elements */
}
' > templates/styles.css

# --- Writing to script.js ---
echo $'// script.js - Enhanced JavaScript for Trading Bot Control Panel

// --- Constants and DOM Elements ---
const startButton = document.getElementById(\'start-button\');
const stopButton = document.getElementById(\'stop-button\');
const logsButton = document.querySelector(\'.btn-group a[href="/logs"]\'); // Select logs button more specifically
const botStatusSpan = document.getElementById(\'bot-status\');
const messageDiv = document.getElementById(\'message\');
const terminalOutputArea = document.getElementById(\'terminal-output-area\');
const tradingSymbolInput = document.getElementById(\'tradingSymbol\');
const symbolValidationMessage = document.getElementById(\'symbol-validation-message\');
const progressBar = document.getElementById(\'progress-bar\');
const progressBarInner = progressBar.querySelector(\'.progress-bar\');

const BOT_STATUS_RUNNING = \'Running\';
const BOT_STATUS_STOPPED = \'Stopped\';
const BOT_STATUS_AWAITING = \'Awaiting Commands\';

const SYMBOL_REGEX = /^[a-zA-Z0-9]+\\/[a-zA-Z0-9]+$/; // Basic symbol regex (e.g., BTC/USDT)
const DATA_FETCH_INTERVAL = 15000; // 15 seconds in milliseconds

// --- Utility Functions ---

/**
 * Appends a line to the terminal output area.
 * @param {string} text - The HTML string to append to the terminal.
 */
function appendToTerminal(text) {
    terminalOutputArea.innerHTML += `<div class="terminal-line">${text}</div>`;
    terminalOutputArea.scrollTop = terminalOutputArea.scrollHeight; // Scroll to bottom
}

/**
 * Displays a message in the message div with appropriate styling.
 * @param {string} message - The message to display.
 * @param {string} type - \'success\', \'warning\', \'danger\', or \'info\'.
 */
function displayMessage(message, type = \'info\') {
    messageDiv.textContent = message;
    messageDiv.className = `alert mt-3 text-center alert-${type}`;
    messageDiv.setAttribute(\'role\', \'alert\'); // Ensure ARIA role is set each time
}

/**
 * Clears the message div.
 */
function clearMessage() {
    messageDiv.textContent = \'\';
    messageDiv.className = \'alert mt-3 text-center\'; // Reset class
    messageDiv.removeAttribute(\'role\'); // Remove ARIA role when no message
}

/**
 * Validates the trading symbol input.
 * @param {string} symbol - The trading symbol string.
 * @returns {boolean} - True if valid, false otherwise.
 */
function validateTradingSymbol(symbol) {
    if (!symbol) {
        symbolValidationMessage.textContent = \'Trading symbol is required.\';
        tradingSymbolInput.classList.add(\'is-invalid\');
        return false;
    }
    if (!SYMBOL_REGEX.test(symbol)) {
        symbolValidationMessage.textContent = \'Invalid symbol format (e.g., BTC/USDT).\';
        tradingSymbolInput.classList.add(\'is-invalid\');
        return false;
    }
    symbolValidationMessage.textContent = \'\'; // Clear message
    tradingSymbolInput.classList.remove(\'is-invalid\');
    return true;
}

/**
 * Shows the progress bar.
 */
function showProgressBar() {
    progressBar.classList.remove(\'d-none\');
}

/**
 * Hides the progress bar.
 */
function hideProgressBar() {
    progressBar.classList.add(\'d-none\');
    progressBarInner.style.width = \'0%\'; // Reset progress
    progressBarInner.setAttribute(\'aria-valuenow\', 0);
}

/**
 * Updates the progress bar percentage.
 * @param {number} percentage - The progress percentage (0-100).
 */
function updateProgressBar(percentage) {
    progressBarInner.style.width = `${percentage}%`;
    progressBarInner.setAttribute(\'aria-valuenow\', percentage);
}


// --- API Interaction Functions ---

/**
 * Starts the trading bot.
 * @param {string} symbol - The trading symbol.
 */
async function startBot(symbol) {
    if (!validateTradingSymbol(symbol)) {
        return; // Stop if symbol is invalid
    }

    showProgressBar();
    clearMessage(); // Clear previous messages

    const formData = new FormData();
    formData.append(\'symbol\', symbol);

    try {
        const response = await fetch(\'/start_bot\', {
            method: \'POST\',
            body: formData, // FormData is automatically handled by fetch
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        hideProgressBar();

        if (data.status === \'success\') {
            botStatusSpan.textContent = BOT_STATUS_RUNNING;
            startButton.disabled = true;
            stopButton.disabled = false;
            logsButton.disabled = false; // Enable logs button when bot starts
            displayMessage(data.message, \'success\');
            appendToTerminal(`<span class="terminal-prompt">Bot:</span> <span class="terminal-output">Bot started successfully for ${symbol}.</span>`);
        } else if (data.status === \'warning\') {
            displayMessage(data.message, \'warning\');
            appendToTerminal(`<span class="terminal-prompt">Warning:</span> <span class="terminal-output">${data.message}</span>`);
        } else {
            displayMessage(data.message, \'danger\');
            appendToTerminal(`<span class="terminal-prompt">Error:</span> <span class="terminal-error">${data.message}</span>`);
        }

    } catch (error) {
        hideProgressBar();
        console.error(\'Error starting bot:\', error);
        displayMessage(\'Failed to start bot. Check console for details.\', \'danger\');
        appendToTerminal(`<span class="terminal-prompt">Error:</span> <span class="terminal-error">Error starting bot: ${error.message}</span>`);
    }
}


/**
 * Stops the trading bot.
 */
async function stopBot() {
    showProgressBar();
    clearMessage();

    try {
        const response = await fetch(\'/stop_bot\', { method: \'POST\' });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        hideProgressBar();

        if (data.status === \'success\') {
            botStatusSpan.textContent = BOT_STATUS_STOPPED;
            startButton.disabled = false;
            stopButton.disabled = true;
            displayMessage(data.message, \'success\');
            appendToTerminal(`<span class="terminal-prompt">Bot:</span> <span class="terminal-output">Bot stopped.</span>`);
        } else if (data.status === \'warning\') {
            displayMessage(data.message, \'warning\');
            appendToTerminal(`<span class="terminal-prompt">Warning:</span> <span class="terminal-output">${data.message}</span>`);
        } else {
            displayMessage(data.message, \'danger\');
            appendToTerminal(`<span class="terminal-prompt">Error:</span> <span class="terminal-error">${data.message}</span>`);
        }

    } catch (error) {
        hideProgressBar();
        console.error(\'Error stopping bot:\', error);
        displayMessage(\'Failed to stop bot. Check console for details.\', \'danger\');
        appendToTerminal(`<span class="terminal-prompt">Error:</span> <span class="terminal-error">Error stopping bot: ${error.message}</span>`);
    }
}


/**
 * Fetches and updates bot data (account balance, orders, PnL).
 */
async function fetchBotData() {
    if (startButton.disabled === false) { // Only fetch data when bot is running
        return;
    }

    try {
        const response = await fetch(\'/account_data\');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === \'success\') {
            let terminalText = "";
            terminalText += `<div class="terminal-line"><span class="terminal-prompt">Balance:</span> <span class="terminal-output">${data.balance} USDT</span></div>`;

            terminalText += `<div class="terminal-line"><span class="terminal-prompt">Open Orders:</span></div>`;
            if (data.orders && data.orders.length > 0) { // Check if orders exist and are not empty
                data.orders.forEach(order => {
                    terminalText += `<div class="terminal-line"><span class="terminal-output">  - [\${order.side.toUpperCase()}] \${order.amount} \${order.symbol} @ \${order.price || \'Market Price\'}</span></div>`;
                });
            } else {
                terminalText += `<div class="terminal-line"><span class="terminal-output">  No open orders.</span></div>`;
            }
            terminalText += `<div class="terminal-line"><span class="terminal-prompt">Position PnL:</span> <span class="terminal-output">${data.pnl}%</span></div>`;
            terminalOutputArea.innerHTML = terminalText; // Update terminal area
        } else {
            appendToTerminal(`<div class="terminal-line"><span class="terminal-error">Error fetching account data: ${data.message}. Check logs.</span></div>`);
            console.error(\'Error fetching account data:\', data.message);
        }

    } catch (error) {
        appendToTerminal(`<div class="terminal-line"><span class="terminal-error">Error fetching account data. Network issue. Check console.</span></div>`);
        console.error(\'Error fetching account data:\', error);
    }
}


// --- Event Listeners ---

startButton.addEventListener(\'click\', () => {
    const symbol = tradingSymbolInput.value.trim();
    startBot(symbol);
});

stopButton.addEventListener(\'click\', stopBot);


// --- Initialization and Periodic Updates ---

/**
 * Initializes the bot control panel.
 */
function initialize() {
    botStatusSpan.textContent = BOT_STATUS_AWAITING;
    startButton.disabled = false; // Enable start button on page load
    stopButton.disabled = true;  // Keep stop button disabled initially
    logsButton.disabled = true;   // Keep logs button disabled initially

    // Fetch initial bot data immediately on load (optional, if needed)
    // fetchBotData(); // Uncomment if you want initial data on load

    // Start periodic data fetching
    setInterval(fetchBotData, DATA_FETCH_INTERVAL);
}


// Call initialize when the script loads
initialize();
' > templates/script.js

echo "Web files (index.html, styles.css, script.js) created in the 'templates' directory."
echo "Make sure these files are in the 'templates' directory, 'app.py', 'trading_bot.py', 'config.yaml', and '.env' are in the project root."
echo "To run the Flask app, use the command: python app.py"
