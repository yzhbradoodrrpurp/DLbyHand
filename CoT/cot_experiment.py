# -*- coding = utf-8 -*-
# @Time: 2025/12/13 15:31
# @Author: Zhihang Yi
# @File: cot_experiment.py
# @Software: PyCharm

from openai import OpenAI
import os
import yaml
import logging


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("experiment.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)


def zero_shot(client, model, questions):
    logger.info("Starting zero-shot processing...")

    logger.info("Loading system prompt for zero-shot...")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    content = config['zero_shot_system_prompt']

    logger.info("Beginning question processing...")
    try:
        for question in questions:
            logger.info(f"Processing question: {question}")
            messages = [
                {'role': 'system', 'content': content},
                {'role': 'user', 'content': f'根据问题直接得到答案：\n\n{question}'}
            ]

            completion = client.chat.completions.create(
                model=model,
                messages=messages
            )

            logger.info('Received response from model.')
            response = completion.choices[0].message.content

            logger.info('Writing response to file...')
            with open('zero_shot_sample.txt', 'a', encoding='utf-8') as f:
                f.write(f"Question:\n{question}\n\nAnswer:\n{response}\n\n{'-'*50}\n\n")

    except Exception as e:
        logger.error(f"Error during zero-shot processing: {e}")
        print(f"Error during zero-shot processing: {e}")


def cot(client, model, questions):
    logger.info("Starting CoT processing...")

    logger.info("Loading system prompt for CoT...")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    content = config['cot_system_prompt']

    logger.info("Beginning question processing...")
    try:
        for question in questions:
            logger.info(f"Processing question: {question}")
            messages = [
                {'role': 'system', 'content': content},
                {'role': 'user', 'content': f'根据问题，一步一步地分析，得到最终的答案：\n\n{question}'}
            ]

            completion = client.chat.completions.create(
                model=model,
                messages=messages
            )

            logger.info('Received response from model.')
            response = completion.choices[0].message.content

            logger.info('Writing response to file...')
            with open('cot_sample.txt', 'a', encoding='utf-8') as f:
                f.write(f"Question:\n{question}\n\nAnswer:\n{response}\n\n{'-'*50}\n\n")

    except Exception as e:
        logger.error(f"Error during CoT processing: {e}")
        print(f"Error during CoT processing: {e}")


if __name__ == "__main__":
    logger.info("Starting CoT experiment...")

    try:
        logger.info("Loading API key...")
        api_key = os.getenv("api_key")
    except Exception as e:
        logger.error(f"Failed to load API key: {e}")
        print(f"Error retrieving API key: {e}")

    logger.info('Loading configuration...')
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_url = config['base_url']

    logger.info('Initializing OpenAI client...')
    client = OpenAI(api_key=api_key, base_url=base_url)

    model = config['model']
    questions = config['questions']

    zero_shot(client, model, questions)
    cot(client, model, questions)





