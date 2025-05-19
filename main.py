import os
import json
import traceback
from time import sleep
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from confluent_kafka import Consumer, Producer

# === Load environment variables ===
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC")
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([KAFKA_BOOTSTRAP_SERVERS, KAFKA_USERNAME, KAFKA_PASSWORD, KAFKA_INPUT_TOPIC, KAFKA_OUTPUT_TOPIC, GROQ_API_KEY]):
    raise Exception("❌ Missing one or more required environment variables.")

# === Kafka Consumer & Producer Configuration ===
consumer_conf = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'group.id': 'groq-sentiment-analyzer',
    'auto.offset.reset': 'earliest',
    'security.protocol': 'SASL_PLAINTEXT',
    'sasl.mechanism': 'SCRAM-SHA-512',
    'sasl.username': KAFKA_USERNAME,
    'sasl.password': KAFKA_PASSWORD
}

consumer = Consumer(consumer_conf)
consumer.subscribe([KAFKA_INPUT_TOPIC])

producer = Producer({
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'security.protocol': 'SASL_PLAINTEXT',
    'sasl.mechanism': 'SCRAM-SHA-512',
    'sasl.username': KAFKA_USERNAME,
    'sasl.password': KAFKA_PASSWORD
})

# === LLM Setup ===
groq_primary_llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY
)

groq_fallback_llm = ChatGroq(
    model_name="mistral-7b-instruct",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY
)

# === Prompt Templates ===
sentiment_prompt = PromptTemplate(
    input_variables=["notes"],
    template="""
Classify the following cold call notes as one of the following sentiment labels:
- Positive
- Neutral / Follow-up Required
- Negative

Notes:
{notes}

Return only the label and reason.
"""
)

action_prompt = PromptTemplate(
    input_variables=["notes", "sentiment"],
    template="""
You are a sales assistant for a Kafka-based platform called Condense.

The following are notes from a recent cold call:
"{notes}"

Sentiment label: {sentiment}

Based on this:
- If the sentiment label is "Positive":
  - Carefully read the caller notes and come up with the sales reachout plan step by step which should be customised based on caller notes. Write ready-to-consume reachout and sales plan.

- If the sentiment label is "Follow-up Required" or "Neutral":
  - Check the caller notes.
  - If there are signs of mild interest, or open-ended comments, come up with the sales reachout plan step by step which should be customised based on caller notes. Write ready-to-consume reachout and sales plan.

- If the sentiment label is "Negative":
  - Do not suggest any follow-up. Just say: "No further action recommended based on negative sentiment."

Respond clearly and helpfully.
"""
)

# === Core Function ===
def classify_and_generate(notes):
    try:
        sentiment = groq_primary_llm.invoke(sentiment_prompt.format(notes=notes)).content.strip()
    except Exception as e:
        print(f"⚠️ Primary sentiment LLM failed: {e}\n🔁 Using fallback.")
        try:
            sentiment = groq_fallback_llm.invoke(sentiment_prompt.format(notes=notes)).content.strip()
        except Exception as e2:
            return {
                "Caller Notes": notes,
                "Sentiment Label": "Unknown",
                "Reason": f"❌ Both LLMs failed for sentiment: {e2}",
                "Next Action Items": "❌ Skipped due to classification error"
            }

    if sentiment.lower().startswith("negative"):
        return {
            "Caller Notes": notes,
            "Sentiment Label": sentiment,
            "Reason": "Negative sentiment detected",
            "Next Action Items": "❌ No action recommended due to negative sentiment."
        }

    prompt = action_prompt.format(notes=notes, sentiment=sentiment)
    try:
        action_items = groq_primary_llm.invoke(prompt).content.strip()
    except Exception as e:
        print(f"⚠️ Primary action LLM failed: {e}\n🔁 Using fallback.")
        try:
            action_items = groq_fallback_llm.invoke(prompt).content.strip()
        except Exception as e2:
            action_items = f"❌ Both LLMs failed for action generation: {e2}"

    return {
        "Caller Notes": notes,
        "Sentiment Label": sentiment,
        "Reason": "Sentiment identified via Groq LLM",
        "Next Action Items": action_items
    }

# === Kafka Loop ===
print(f"📡 Listening to Kafka topic: {KAFKA_INPUT_TOPIC}")
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"⚠️ Kafka error: {msg.error()}")
            continue

        try:
            payload = json.loads(msg.value().decode("utf-8"))
            notes = payload.get("comments", "").strip()
            if not notes:
                print("⚠️ Skipping empty comment field")
                continue

            result = classify_and_generate(notes)
            payload["result"] = result

            enriched = json.dumps(payload).encode("utf-8")
            producer.produce(KAFKA_OUTPUT_TOPIC, enriched)
            producer.flush()
            print(f"✅ Published enriched message to {KAFKA_OUTPUT_TOPIC}")

        except Exception as e:
            print("[PROCESSING ERROR]", e)
            traceback.print_exc()

        sleep(0.25)

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")
finally:
    consumer.close()
