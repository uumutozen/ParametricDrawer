from pathlib import Path
from openai import OpenAI

client = OpenAI()
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Once Upon a Time… Deep in the forest, there lived a very fast rabbit. This rabbit would run around the forest all day, thinking that his speed was unmatched. He constantly bragged to the other animals about how fast he was and claimed that he could outrun anyone. One day, the rabbit proudly shouted, “There is no one in this forest faster than me! Who dares to race against me?” Hearing this, an old and wise tortoise quietly approached the rabbit. The tortoise said, “Brother Rabbit, I would like to race against you.” The rabbit burst into laughter at this offer because the tortoise was known for being very slow. “Brother Tortoise, do you really think you can beat me?” the rabbit asked. The tortoise calmly replied, “We won’t know until we try.” And so, all the animals in the forest gathered to watch the race. As soon as the race started, the rabbit dashed forward at full speed, while the tortoise moved slowly but steadily. Seeing that the tortoise was far behind, the rabbit decided to rest under a tree. “He will never catch up to me,” he thought. However, the tortoise kept moving forward without stopping—slowly but determinedly. While the rabbit was in deep sleep, the tortoise quietly passed him. When the rabbit finally woke up, he saw that the tortoise was far ahead. He ran as fast as he could, but it was too late! The tortoise had already reached the finish line and won the race. The rabbit accepted the tortoise’s victory and learned an important lesson: speed is not everything—determination and perseverance matter the most. From that day on, all the animals in the forest admired the tortoise for his patience and strong spirit.",
)
response.stream_to_file(speech_file_path)