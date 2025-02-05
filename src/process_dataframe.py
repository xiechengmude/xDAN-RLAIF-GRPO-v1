import pandas as pd
import json
import logging
from typing import Dict, List, Union, Optional
from collections import Counter

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 角色映射
ROLE_MAPPING = {
    'user': 'human',
    'assistant': 'gpt'
}

def analyze_dataset(df: pd.DataFrame) -> Dict:
    """分析数据集中的问题."""
    stats = {
        'total_rows': len(df),
        'null_system': df['system'].isna().sum(),
        'null_conversations': df['conversations'].isna().sum(),
        'empty_conversations': 0,
        'invalid_json': 0,
        'role_counts': Counter(),
        'missing_role': 0,
        'missing_value': 0,
        'none_value': 0
    }
    
    for idx, row in df.iterrows():
        # 检查conversations
        conversations = row['conversations']
        if isinstance(conversations, float) and pd.isna(conversations):  # 正确处理NA值
            continue
            
        try:
            convs = json.loads(conversations) if isinstance(conversations, str) else conversations
            if not convs:
                stats['empty_conversations'] += 1
                continue
                
            for conv in convs:
                if not isinstance(conv, dict):
                    continue
                    
                # 统计角色分布
                role = conv.get('from')
                if role is None:
                    stats['missing_role'] += 1
                else:
                    stats['role_counts'][role] += 1
                
                # 检查value字段
                if 'value' not in conv:
                    stats['missing_value'] += 1
                elif conv['value'] is None:
                    stats['none_value'] += 1
                    
        except json.JSONDecodeError:
            stats['invalid_json'] += 1
            logger.warning(f"JSON decode error at row {idx}")
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            stats['invalid_json'] += 1
            
    return stats

def print_dataset_analysis(stats: Dict):
    """打印数据集分析结果."""
    logger.info("=== 数据集分析结果 ===")
    logger.info(f"总行数: {stats['total_rows']}")
    logger.info(f"system列空值数: {stats['null_system']}")
    logger.info(f"conversations列空值数: {stats['null_conversations']}")
    logger.info(f"空对话数: {stats['empty_conversations']}")
    logger.info(f"JSON解析错误数: {stats['invalid_json']}")
    logger.info(f"缺失角色数: {stats['missing_role']}")
    logger.info(f"缺失value字段数: {stats['missing_value']}")
    logger.info(f"value为None的数量: {stats['none_value']}")
    logger.info("\n角色分布:")
    for role, count in stats['role_counts'].items():
        logger.info(f"  {role}: {count}")

def process_system_prompt(text: Optional[str]) -> str:
    """Replace the system prompt with think and answer tags format."""
    if not isinstance(text, str):
        logger.warning(f"Invalid system prompt type: {type(text)}, expected string")
        return "" if text is None else str(text)
    
    return text.replace(
        '<|begin_of_thought|>',
        '<think>'
    ).replace(
        '<|end_of_thought|>',
        '</think>'
    ).replace(
        '<|begin_of_solution|>',
        '<answer>'
    ).replace(
        '<|end_of_solution|>',
        '</answer>'
    )

def extract_content_between_tags(text: str, start_tag: str, end_tag: str) -> str:
    """Extract content between given tags safely."""
    try:
        start_idx = text.index(start_tag) + len(start_tag)
        end_idx = text.index(end_tag, start_idx)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return ""

def validate_conversation(conv: Dict) -> bool:
    """Validate conversation format and structure."""
    if not isinstance(conv, dict):
        logger.warning(f"Invalid conversation format: {conv}")
        return False
    
    if 'from' not in conv:
        logger.warning(f"Missing 'from' field in conversation: {conv}")
        return False
        
    if 'value' not in conv:
        logger.warning(f"Missing 'value' field in conversation: {conv}")
        return False
    
    # 检查并记录None值
    if conv['value'] is None:
        logger.warning(f"Found None value in conversation with role: {conv.get('from')}")
        conv['value'] = ""  # 将None替换为空字符串
    
    if not isinstance(conv['value'], str):
        conv['value'] = str(conv['value'])
        logger.warning(f"Converting non-string value to string: {conv['value']}")
    
    return True

def process_assistant_response(value: str) -> str:
    """Process assistant's response with proper tag handling."""
    # 提取思考部分
    thought = extract_content_between_tags(
        value, 
        '<|begin_of_thought|>', 
        '<|end_of_thought|>'
    )
    
    # 提取答案部分
    solution = extract_content_between_tags(
        value, 
        '<|begin_of_solution|>', 
        '<|end_of_solution|>'
    )
    
    # 构建新的响应格式
    new_value = []
    if thought:
        new_value.append(f"<think>{thought}</think>")
    if solution:
        new_value.append(f"<answer>{solution}</answer>")
    
    return "\n".join(new_value) if new_value else value

def process_conversation_value(conv: Dict) -> str:
    """Process a single conversation value based on the role."""
    if not isinstance(conv.get('value'), str):
        conv['value'] = str(conv['value']) if conv['value'] is not None else ""
        logger.warning(f"Converting non-string value to string: {conv['value']}")
    
    value = conv['value']
    
    # 只处理 gpt 的回复
    if conv.get('from') == 'assistant' or conv.get('from') == 'gpt':
        return process_assistant_response(value)
    
    return value

def map_role(role: str) -> str:
    """Map old role names to new ones."""
    return ROLE_MAPPING.get(role, role)

def process_conversations(conversations: Union[str, List[Dict]]) -> List[Dict]:
    """Process the conversations list to update the format for multi-turn dialogues."""
    try:
        if isinstance(conversations, str):
            try:
                conv_list = json.loads(conversations)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse conversation JSON: {e}")
                return []
        else:
            conv_list = conversations

        if not isinstance(conv_list, list):
            logger.error(f"Invalid conversations format: {type(conv_list)}")
            return []
            
        if not conv_list:
            logger.warning("Empty conversation list")
            return []

        processed_conv = []
        for i, conv in enumerate(conv_list):
            if not validate_conversation(conv):
                logger.warning(f"Invalid conversation at position {i}")
                continue
                
            processed_value = process_conversation_value(conv)
            processed_conv.append({
                'from': map_role(conv['from']),
                'value': processed_value
            })

        if not processed_conv:
            logger.warning("All conversations were invalid and filtered out")
            
        return processed_conv

    except Exception as e:
        logger.error(f"Error processing conversations: {e}")
        return []

def main():
    try:
        # Load your dataframe
        input_path = 'your_dataframe.csv'  # Replace with your actual file path
        logger.info(f"Loading dataframe from {input_path}")
        df_dt_openThought = pd.read_csv(input_path)
        
        # 分析数据集
        logger.info("Analyzing dataset...")
        stats = analyze_dataset(df_dt_openThought)
        print_dataset_analysis(stats)
        
        # Process system column
        logger.info("Processing system prompts...")
        df_dt_openThought['system'] = df_dt_openThought['system'].apply(process_system_prompt)
        
        # Process conversations column
        logger.info("Processing conversations...")
        df_dt_openThought['conversations'] = df_dt_openThought['conversations'].apply(process_conversations)
        
        # Remove any rows where conversations processing failed
        original_len = len(df_dt_openThought)
        df_dt_openThought = df_dt_openThought[df_dt_openThought['conversations'].apply(len) > 0]
        filtered_len = len(df_dt_openThought)
        
        # Save the processed dataframe
        output_path = 'processed_dataframe.csv'
        df_dt_openThought.to_csv(output_path, index=False)
        logger.info(f"DataFrame processing completed! Saved to {output_path}")
        
        # 输出处理统计信息
        logger.info(f"Total original rows: {original_len}")
        logger.info(f"Rows after filtering: {filtered_len}")
        logger.info(f"Removed rows: {original_len - filtered_len}")
        
    except Exception as e:
        logger.error(f"Error processing dataframe: {e}")
        raise

if __name__ == "__main__":
    main()
