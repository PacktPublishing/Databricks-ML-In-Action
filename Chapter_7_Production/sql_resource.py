SQL_INSTRUCTION = """You are a Data Analyst and SQL expert, here to build a SQL Query in response to a business user's prompt. Please do not make assumptions if the user has given you an ambiguous request. Getting the SQL Query correct is extremely important. If you do not have the information you need to write the correct SQL Statement, request additional information. 

Your responses are very limited and must be returned in a strict format of "[response-type]|[response]".

Table of Allowed Responses:
response-type|response-value|description
list|tables|This will request a list of tables
define|[table-name]|This will retrieve the table definition for the provided table names. Replace [table-name] with a comma separated list of table names
sample|[table-name]|This will retrieve sample records for the provided table names. Replace [table-name] with a comma separated list of table names
Q|[question]|Allows you to ask the user a clarifying question. Replace [question] with your clarifying question. Only use this response as a last resort if you need information beyond the other responses.
SQL|[SQL]|Use this to return the final SQL Statement that answers the user's original prompt. [SQL] should be replaced with the generated SQL. This response will end the chat conversation.

"""
"""
Examples of Allowed Responses:
1. Table Listing: To request a list of tables that are available for the SQL query, return:
list|tables

2. Table Definition: To get the definition of a table (to see the columns and related information), return:
define|[table_name]
replacing [table_name] with one of the tables listed above. Alternatively, you can request multiple tables by providing a comma separated list.

3. Sample Data: To get sample records of a table, return:
sample|[table_name]
replacing [table_name] with one of the tables listed above. Alternatively, you can request multiple tables by providing a comma separated list.

4. Question to User: To ask a clarifying question, return:
Q|[question]
replacing [question] with the question or summary of information you need.
Note: Only use this response as a last resort, if the prior three responses can't answer your question.

5. Answer: To return the final SQL statement, return:
SQL|[SQL]
replacing [SQL] with the SQL statement that will address the user's original prompt. The tables should be aliased and the columns should be fully declared.
"""

INVALID_RESPONSE = 'I did not understand your response. Please make sure you are replying in the form of [response-type]|[response] per your instructions.'

FINAL_INSTRUCTION = """
Given this data below:

{data}

And the context provided below:

{context}

Provide an answer to the following prompt:

{prompt}

"""

CATALOG = 'main'
SCHEMA = 'default'
conversation = []
conversation_id = ""
original_prompt = ''

from databricks.sdk.runtime import *
import openai
import os
import uuid
import copy

OPENAI_API_KEY= "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

def setDataLocation(catalog, schema):
  global CATALOG
  global SCHEMA
  CATALOG = catalog
  SCHEMA = schema
  spark.sql(f'use catalog {CATALOG};')
  spark.sql(f'use schema {SCHEMA};')

def getTableList():
  sql = f"""
        select table_name, comment 
        from system.information_schema.tables
        where table_catalog = '{CATALOG}' and table_schema = '{SCHEMA}'
        """
  #print(sql)
  tables = spark.sql(sql)
  return f'List of tables: \ntable_index' + tables.toPandas().to_csv(sep='|')

def table_def(df, table): 
  table_schema = df.drop('table_catalog', 'table_schema', 'table_name', 'ordinal_position', 'character_octet_length',	'numeric_precision',	'numeric_precision_radix',	'numeric_scale',	'datetime_precision',	'interval_type',	'interval_precision', 'identity_start',	'identity_increment',	'identity_maximum',	'identity_minimum',	'identity_cycle', 'is_system_time_period_start',	'is_system_time_period_end',	'system_time_period_timestamp_generation',	'is_updatable').where(f'table_name = "{table}"')

  return f'Table Schema for {table}: \ncolumn_index' + table_schema.toPandas().to_csv(sep='|')

def getTableDefinition(tables):
  tables_arr = tables.replace(' ', '').split(',')
  tables = "','".join(tables_arr)

  table_schemas = spark.sql(f"""
      with constraints as (
        select k.*, cs.constraint_type, u.table_catalog referential_table_catalog, u.table_schema referential_table_schema, u.table_name referential_table_name
        from system.information_schema.key_column_usage k
          inner join system.information_schema.table_constraints cs on k.constraint_catalog = cs.constraint_catalog and k.constraint_schema = cs.constraint_schema and k.constraint_name = cs.constraint_name
          left outer join (select distinct constraint_catalog, constraint_schema, constraint_name, table_catalog, table_schema, table_name from system.information_schema.constraint_column_usage) u on k.constraint_catalog = u.constraint_catalog and k.constraint_schema = u.constraint_schema and k.constraint_name = u.constraint_name and cs.constraint_type = 'FOREIGN KEY'
        where k.table_catalog = '{CATALOG}' and k.table_schema = '{SCHEMA}' and k.table_name in ('{tables}')
      )
      select c.*, cs.constraint_name, cs.ordinal_position constraint_ordinal_position, cs.constraint_type, cs.referential_table_catalog, cs.referential_table_schema, cs.referential_table_name
      from system.information_schema.columns c 
        left outer join constraints cs on c.table_catalog = cs.table_catalog and c.table_schema = cs.table_schema and c.table_name = cs.table_name and c.column_name = cs.column_name
      where c.table_catalog = '{CATALOG}' and c.table_schema = '{SCHEMA}'and c.table_name in ('{tables}')
      order by table_name, ordinal_position;""")
  
  ret = ""
  for tbl in tables_arr:
    ret += table_def(table_schemas.where(f'table_name = "{tbl}"'), tbl) + "\n\n"

  return ret

def table_sample(table): 
  records = spark.sql(f'select * from {table} limit 2')

  return f'Sample records for {table}: \ncolumn_index' + records.toPandas().to_csv(sep='|')

def getTableSampleRecords(tables):
  tables_arr = tables.replace(' ', '').split(',')

  ret = ""
  for tbl in tables_arr:
    ret += table_sample(tbl) + "\n\n"
    
  return ret

def submit_conversation(convo_array):
  #print(convo_array)
  completion = openai.ChatCompletion.create(
    #model="gpt-4",
    model="gpt-3.5-turbo",
    messages=convo_array
  )

  return completion.choices[0].message.content


def log_conversation(convo_type, convo_array):
  row = 1
  convo = copy.deepcopy(convo_array)
  for line in convo:
    line['conversation_id'] = conversation_id
    line['conversation_type'] = convo_type
    line['ordinal_position'] = row
    row += 1

  df = spark.createDataFrame(convo)
  df.write.mode('append').saveAsTable(f'robert_mosley.sql_ai.conversation_lines')
  #print(f'{convo_type}: {convo_array}')

def processSQL(sql):
  df = spark.sql(sql)
  ret = []
  ret.append(buildPromptItem('system', "You are a Data Analyst. Use Data and additional Context (if provided) to respond to the prompt."))
  ret.append(buildPromptItem('user', FINAL_INSTRUCTION.format(data=df.toPandas().to_csv(sep='|'), context="", prompt=original_prompt)))
  resp = submit_conversation(ret)
  ret.append(buildPromptItem('assistant', resp))
  log_conversation('SQL_RESPONSE_INTERPRETATION', ret)
  return resp


#{"role": "user", "content": "Who won the world series in 2020?"},
def buildPromptItem(role, content):
  #system, assistant, user
  return {"role": role, "content": content}

def process_conversation():
  #print(f'process: {conversation}, original prompt {original_prompt}')
  while 1==1:
    resp = submit_conversation(conversation)
    conversation.append(buildPromptItem('assistant', resp))
    
    resp_type = resp.split('|')[0]
    resp_value = resp.split('|')[1]
    
    if resp_type == 'list': #list tables
      print('LLM Requesting List of Tables...')
      conversation.append(buildPromptItem('user', getTableList()))
      continue
    elif resp_type == 'define': #define table schema
      print('LLM Requesting Table Definitions...')
      conversation.append(buildPromptItem('user', getTableDefinition(resp_value)))
      continue
    elif resp_type == 'sample': #get sample records for table
      print('LLM Requesting Data Samples...')
      conversation.append(buildPromptItem('user', getTableSampleRecords(resp_value)))
      continue
    elif resp_type == 'Q': #Question for User
      print(resp_value)
      return resp_value
    elif resp_type == 'SQL': #SQL for getting the final answer
      print('LLM Querying Data...')
      log_conversation('SQL_ANALYST', conversation)
      answer = processSQL(resp_value)
      print(answer)
      return answer
    else:
      print('LLM Gave an Invalid Response...')
      conversation.append(buildPromptItem('user', INVALID_RESPONSE))
'''
    match resp_type:
      case 'list': #list tables
        conversation.append('user', getTableList(resp_value))
        continue
      case 'define': #get table schema
        conversation.append('user', getTableDefinition(resp_value))
        continue
      case 'sample':
        conversation.append('user', getTableSampleRecords(resp_value))
        continue
      case 'Q':
        print(resp_value)
        return resp_value
      case 'SQL':
        log_conversation('SQL_ANALYST', conversation)
        answer = processSQL(resp_value)
        print(answer)
        return answer
        break
'''
      
          
  #test for SQL

def startConversation(prompt):
  global original_prompt 
  original_prompt = prompt
  global conversation
  conversation = []
  global conversation_id
  conversation_id = str(uuid.uuid4())
  conversation.append(buildPromptItem('system', SQL_INSTRUCTION))
  conversation.append(buildPromptItem('user', prompt))
  #print(f'starting convo: {conversation}')
  process_conversation()

def continueConversation(context):
  conversation.append(buildPromptItem('user', context))
  process_conversation()

