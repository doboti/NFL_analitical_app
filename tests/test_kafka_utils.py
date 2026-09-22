"""
Unit teszt a consume_forever hibatűrő üzenetfeldolgozó ciklusra.

Ez a logika került bevezetésre azután, hogy kiderült: egyetlen hibás Kafka
üzenet vagy handler-kivétel az egész mikroszolgáltatást leállította (lásd a
"könnyen összeomlik" hibajegyet). A teszt egy hamis (fake) consumer-t használ,
NEM igényel valódi Kafka brókert.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

import kafka_utils  # noqa: E402
from kafka_utils import consume_forever  # noqa: E402


class FakeMessage(SimpleNamespace):
    pass


def make_messages(n):
    return [FakeMessage(value=i, topic="t", offset=i) for i in range(n)]


def test_all_messages_processed_when_no_errors():
    messages = make_messages(5)
    seen = []
    consume_forever(iter(messages), handler=lambda msg: seen.append(msg.value), service_name="test")
    assert seen == [0, 1, 2, 3, 4]


def test_handler_exception_does_not_stop_processing():
    messages = make_messages(5)
    seen = []

    def handler(msg):
        if msg.value == 2:
            raise ValueError("szimulált hibás üzenet")
        seen.append(msg.value)

    consume_forever(iter(messages), handler=handler, service_name="test")
    # Minden üzenetet megkapott a handler, csak a hibás (2) nem került be a listába,
    # de a feldolgozás nem állt le utána.
    assert seen == [0, 1, 3, 4]


def test_iterator_error_does_not_propagate_and_processing_stops_cleanly():
    """Egy generátorból érkező kivétel (pl. átmeneti Kafka kapcsolati hiba) NEM
    terjed tovább (nem öli meg a szolgáltatást) - a ciklus naplózza, vár, majd
    a kimerült generátor StopIteration-jével rendben leáll."""
    def flaky_iterator():
        yield FakeMessage(value=0, topic="t", offset=0)
        raise RuntimeError("szimulált Kafka kapcsolati hiba")

    seen = []
    with patch.object(kafka_utils.time, "sleep", return_value=None):
        consume_forever(flaky_iterator(), handler=lambda msg: seen.append(msg.value), service_name="test")
    assert seen == [0]
