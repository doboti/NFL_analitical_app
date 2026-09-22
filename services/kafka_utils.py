"""Közös Kafka producer/consumer segédfüggvények a mikroszolgáltatásokhoz."""
import json
import os
import time

from kafka import KafkaConsumer, KafkaProducer

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

TOPIC_RAW_FRAMES = "nfl.raw.frames"
TOPIC_RAW_AUDIO = "nfl.raw.audio"
TOPIC_METADATA = "nfl.metadata"
TOPIC_DETECTIONS = "nfl.detections"
TOPIC_WINPROB = "nfl.winprob"
TOPIC_HIGHLIGHTS = "nfl.highlights"


def wait_for_kafka(max_retries: int = 60, delay: float = 3.0):
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS)
            producer.close()
            return
        except Exception as exc:
            print(f"[kafka_utils] Kafka még nem elérhető ({attempt+1}/{max_retries}): {exc}")
            time.sleep(delay)
    raise RuntimeError("Kafka nem elérhető a megadott próbálkozások után.")


def make_producer() -> KafkaProducer:
    wait_for_kafka()
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_request_size=20 * 1024 * 1024,
    )


def make_consumer(topic: str, group_id: str, from_beginning: bool = True) -> KafkaConsumer:
    wait_for_kafka()
    return KafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=group_id,
        auto_offset_reset="earliest" if from_beginning else "latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        fetch_max_bytes=20 * 1024 * 1024,
        max_partition_fetch_bytes=20 * 1024 * 1024,
    )


def consume_forever(consumer, handler, service_name: str = "service"):
    """Feldolgozza a consumer üzeneteit a handler(msg) hívással, de egy hibás
    üzenet vagy átmeneti Kafka-hiba miatt SOHA nem áll le a folyamat - csak
    naplózza a hibát és megy tovább a következő üzenetre. Ez nélkül egyetlen
    váratlan kivétel (rossz JSON, hiányzó mező, dekódolási hiba egy sérült
    frame-en, stb.) leállítaná az egész konténert, amit a docker-compose
    restart policy-ja ugyan újraindítana, de a köztes állapot elveszne és a
    fogyasztói csoport pozíciója szükségtelenül ugrálna."""
    iterator = iter(consumer)
    while True:
        try:
            msg = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            print(f"[{service_name}] Kafka olvasási hiba, folytatás 2s után: {exc}")
            time.sleep(2)
            continue

        try:
            handler(msg)
        except Exception as exc:
            print(f"[{service_name}] Üzenetfeldolgozási hiba (kihagyva, [{msg.topic}] "
                  f"offset={msg.offset}): {exc}")
