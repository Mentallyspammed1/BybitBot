#!/usr/bin/env bash

# Configuration
PHONE_NUMBER="6364866381"
LOG_FILE="DOTUSDT_20250316_181455.log"
declare -A METRICS ALERTS

# Efficient metric extraction using AWK
extract_metrics() {
    awk '
    function parse_value(str,   parts) {
        split(str, parts, /[=:\(\)]/)
        return parts[2]
    }
    {
        delete metrics
        for (i=1; i<=NF; i++) {
            if ($i == "High:") metrics["prev_high"] = $(i+1)
            if ($i == "Low:") metrics["prev_low"] = $(i+1)
            if ($i == "Close:") metrics["prev_close"] = $(i+1)
            if ($i == "Current") metrics["current_price"] = $(i+2)
            if ($i == "StochRSI" && $(i+1) == "%K:") metrics["stoch_k"] = $(i+2)
            if ($i == "StochRSI" && $(i+1) == "%D:") metrics["stoch_d"] = $(i+2)
            if ($i == "Momentum:") metrics["momentum"] = $(i+1)
            if ($i == "MA200:") metrics["ma_200"] = $(i+1)
            if ($i == "ATR:") metrics["atr"] = $(i+1)
            if ($i == "MACD:") {
                metrics["macd"] = parse_value($(i+1))
                metrics["signal_line"] = parse_value($(i+2))
            }
            if ($i == "RSI" && $(i+1) == "20:") metrics["rsi_20"] = parse_value($(i+3))
            if ($i == "RSI" && $(i+1) == "100:") metrics["rsi_100"] = parse_value($(i+3))
            if ($i == "Vol:") {
                split($0, vol, /[|]/)
                metrics["volume1"] = vol[1]
                metrics["volume2"] = vol[2]
                metrics["volume3"] = vol[3]
            }
            if ($i == "OBV:") metrics["obv"] = $(i+1)
        }
        for (key in metrics)
            printf "%s=%s\n", key, metrics[key]
    }' <<< "$1"
}

# Efficient math calculations using BC
batch_calcs() {
    bc -l <<< "
        scale=4
        pp=($1 + $2 + $3)/3
        range=$1 - $2
        s1=pp - (range * 0.382)
        s2=pp - (range * 0.618)
        r1=pp + (range * 0.382)
        r2=pp + (range * 0.618)
        atr_half=$4 * 0.5
        print pp, s1, s2, r1, r2, atr_half
    "
}

# Real-time processing with enhanced monitoring
process_line() {
    while IFS='=' read -r key value; do
        METRICS["$key"]="$value"
    done < <(extract_metrics "$1")

    # Validate required metrics
    required=(
        prev_high prev_low prev_close current_price stoch_k stoch_d momentum 
        ma_200 atr macd signal_line rsi_20 rsi_100 volume1 volume2 volume3 obv
    )
    for key in "${required[@]}"; do
        [[ -z "${METRICS[$key]}" ]] && return 1
    done

    # Batch calculate pivot points and ATR
    read -r pp s1 s2 r1 r2 atr_half <<< $(batch_calcs \
        "${METRICS[prev_high]}" \
        "${METRICS[prev_low]}" \
        "${METRICS[prev_close]}" \
        "${METRICS[atr]}"
    )

    # Price proximity calculations
    current_price="${METRICS[current_price]}"
    price_proximity() {
        echo "scale=2; ($current_price - $1)/($atr_half)*100" | bc
    }
    proximity_s1=$(price_proximity "$s1")
    proximity_s2=$(price_proximity "$s2")
    proximity_r1=$(price_proximity "$r1")
    proximity_r2=$(price_proximity "$r2")

    # Technical conditions
    CONDITIONS=(
        "price_above_ma=$(bc <<< "${METRICS[current_price]} > ${METRICS[ma_200]}")"
        "stoch_buy=$(bc <<< "${METRICS[stoch_k]} < 20 && ${METRICS[stoch_k]} > ${METRICS[stoch_d]}")"
        "stoch_sell=$(bc <<< "${METRICS[stoch_k]} > 80 && ${METRICS[stoch_k]} < ${METRICS[stoch_d]}")"
        "macd_cross=$(bc <<< "${METRICS[macd]} - ${METRICS[signal_line]}")"
        "rsi_cross=$(bc <<< "${METRICS[rsi_20]} - ${METRICS[rsi_100]}")"
        "volume_trend=$(bc <<< "${METRICS[volume3]} > ${METRICS[volume2]} && ${METRICS[volume2]} > ${METRICS[volume1]}")"
    )

    # Generate alerts for all conditions
    ALERTS=()
    [[ "${CONDITIONS[price_above_ma]}" -eq 1 ]] && ALERTS+=("Bullish Trend")
    [[ "${CONDITIONS[stoch_buy]}" -eq 1 ]] && ALERTS+=("Stochastic Buy Signal")
    [[ $(bc <<< "$macd_cross > 0") -eq 1 ]] && ALERTS+=("MACD Bullish")
    [[ $(bc <<< "$rsi_cross > 0") -eq 1 ]] && ALERTS+=("RSI 20 > 100")
    [[ "${CONDITIONS[volume_trend]}" -eq 1 ]] && ALERTS+=("Volume Increasing")

    # Enhanced SMS alert with all conditions
    sms_message="$(
        printf "LINK/USDT Alert\nPrice: %s\n" "${METRICS[current_price]}"
        printf "Key Levels:\n- S1: %s (%s%%)\n- R1: %s (%s%%)\n" \
            "$s1" "${proximity_s1%%.*}" "$r1" "${proximity_r1%%.*}"
        printf "Conditions:\n- %s\n" "${ALERTS[@]}"
        printf "Indicators:\nRSI20: %s\nRSI100: %s\nVWAP: %s" \
            "${METRICS[rsi_20]}" "${METRICS[rsi_100]}" "$vwap"
    )"

    termux-sms-send -n "$PHONE_NUMBER" "$sms_message" 2>/dev/null
    echo "$sms_message"
}

# Main processing loop with buffer control
tail -n 100 -f "$LOG_FILE" | while read -r line; do
    process_line "$line"
    # Memory optimization for long-running processes
    unset METRICS ALERTS CONDITIONS
done
