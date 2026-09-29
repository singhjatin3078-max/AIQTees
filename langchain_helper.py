import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain.utilities import SQLDatabase
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

from google import genai
from google.genai import types, errors

from few_shots import few_shots

# Load environment variables from .env
load_dotenv()

# Global caches for embeddings and vectorstore to prevent reloading on each query
_vectorstore_cache = None
_embeddings_cache = None

# SQL Safety: Disallow destructive SQL statements
FORBIDDEN_KEYWORDS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bINSERT\b",
    r"\bCREATE\b",
    r"\bREPLACE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bEXEC\b",
    r"\bEXECUTE\b",
    r"\bCALL\b",
]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ──────────────────────────────────────────────
# DOMAIN VALIDATION
# ──────────────────────────────────────────────

class DomainError(Exception):
    """Raised when the user question is outside the T-shirt inventory domain."""
    pass


# Terms that indicate a valid T-shirt inventory question.
# Any question that contains at least one of these (whole-word, case-insensitive)
# is considered in-domain for the product-type axis.
ALLOWED_PRODUCT_TERMS = [
    r"\bt[-\s]?shirts?\b",
    r"\btees?\b",
    r"\bshirts?\b",
    r"\btops?\b",
    r"\binventory\b",
    r"\bstock\b",
    r"\bprice\b",
    r"\bprices\b",
    r"\bdiscount\b",
    r"\bdiscounts\b",
    r"\bsales?\b",
    r"\brevenue\b",
    r"\bcolor\b",
    r"\bcolour\b",
    r"\bsize\b",
    r"\bbrand\b",
    r"\bquantity\b",
]

# Terms that explicitly indicate a NON-T-shirt product.
# These are checked FIRST. If any of these match, the question is rejected
# regardless of other words in the sentence.
OUT_OF_DOMAIN_TERMS = [
    r"\bshoes?\b",
    r"\bsneakers?\b",
    r"\bboots?\b",
    r"\bsandals?\b",
    r"\bjeans?\b",
    r"\btrousers?\b",
    r"\bpants?\b",
    r"\bjackets?\b",
    r"\bcoats?\b",
    r"\bhoodies?\b",
    r"\bsweaters?\b",
    r"\bsweatshirts?\b",
    r"\bdresses?\b",
    r"\bskirts?\b",
    r"\bsocks?\b",
    r"\bgloves?\b",
    r"\bhats?\b",
    r"\bcaps?\b",
    r"\bscarves?\b",
    r"\bbags?\b",
    r"\bbackpacks?\b",
    r"\bpurses?\b",
    r"\bwallets?\b",
    r"\blaptops?\b",
    r"\bphones?\b",
    r"\bmobiles?\b",
    r"\bsmartphones?\b",
    r"\btablets?\b",
    r"\bcomputers?\b",
    r"\bheadphones?\b",
    r"\bearphones?\b",
    r"\bwatches?\b",
    r"\bjewelry\b",
    r"\bjewellery\b",
    r"\bnecklaces?\b",
    r"\bbracelets?\b",
    r"\brings?\b",
    r"\bcars?\b",
    r"\bbikes?\b",
    r"\bvehicles?\b",
    r"\bfurniture\b",
    r"\bchairs?\b",
    r"\btables?\b",
    r"\bfood\b",
    r"\bbooks?\b",
    r"\bperfumes?\b",
    r"\bdeodorants?\b",
]

# Queries that consist solely of brand/attribute lookups with no product noun
# should also be validated against the allowed-product list.
# These brand names are present in the DB — they are valid ONLY in t-shirt context.
KNOWN_BRANDS = [
    r"\bnike\b",
    r"\badidas\b",
    r"\blevi'?s?\b",
    r"\bvan\s+heusen\b",
]


def validate_domain(question: str) -> None:
    """
    Deterministic domain guard. Raises DomainError if:
      (a) the question mentions an explicit out-of-domain product (e.g. shoes, laptops), OR
      (b) the question does not mention any recognised T-shirt term AND does not appear
          to be asking about inventory attributes (stock, price, discount, etc.).

    The check is intentionally conservative on (b): if we cannot find a clear
    T-shirt context word we still let the query through when it ONLY mentions
    brand names or pure inventory attributes — these are inherently t-shirt context
    in this application.  We block only when a concrete non-t-shirt product noun appears.
    """
    q = question.strip()
    if not q:
        raise DomainError("Sorry, I can only answer questions related to the T-shirt inventory.")

    # Step 1 — Hard reject: any out-of-domain product term detected
    for pattern in OUT_OF_DOMAIN_TERMS:
        if re.search(pattern, q, re.IGNORECASE):
            matched = re.search(pattern, q, re.IGNORECASE).group(0)
            raise DomainError(
                f"Sorry, I can only answer questions related to the T-shirt inventory. "
                f"'{matched}' is not part of this database."
            )

    # Step 2 — Soft check: question contains at least one in-domain product/attribute word
    # OR it only contains brand names / generic inventory phrasing (which is fine).
    has_allowed_term = any(
        re.search(p, q, re.IGNORECASE) for p in ALLOWED_PRODUCT_TERMS
    )
    has_known_brand = any(
        re.search(p, q, re.IGNORECASE) for p in KNOWN_BRANDS
    )

    if not has_allowed_term and not has_known_brand:
        raise DomainError(
            "Sorry, I can only answer questions related to the T-shirt inventory. "
            "Please ask about T-shirt stock, prices, brands, sizes, colors, or discounts."
        )


def get_example_selector(k: int = 2):
    """Initializes or retrieves cached FAISS vectorstore for few-shot semantic retrieval."""
    global _vectorstore_cache, _embeddings_cache
    try:
        if _embeddings_cache is None:
            _embeddings_cache = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        if _vectorstore_cache is None:
            to_vectorize = [" ".join(example.values()) for example in few_shots]
            _vectorstore_cache = FAISS.from_texts(to_vectorize, _embeddings_cache, metadatas=few_shots)
        
        return SemanticSimilarityExampleSelector(
            vectorstore=_vectorstore_cache,
            k=k,
        )
    except Exception as e:
        raise RuntimeError(f"Semantic retrieval initialization failed: {e}")


def sanitize_and_extract_sql(raw_output: str) -> str:
    """Extract clean SQL query from model output, stripping code fences, markdown, and preambles."""
    sql = raw_output.strip()
    
    # Check for markdown code blocks (e.g. ```sql ... ```)
    fence_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, re.DOTALL | re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()
    
    # Strip any 'SQLQuery:' or 'SQL Query:' prefix
    prefix_match = re.match(r"^sql\s*query\s*:\s*", sql, re.IGNORECASE)
    if prefix_match:
        sql = sql[prefix_match.end():].strip()
        
    # Strip any trailing 'SQLResult:' or other sections if present
    for marker in ["\nsqlresult:", "\nanswer:", "\nquestion:"]:
        idx = sql.lower().find(marker)
        if idx != -1:
            sql = sql[:idx].strip()
            
    # Remove surrounding quotes or backticks if wrapped
    sql = sql.strip().rstrip(";")
    return sql


def validate_sql(sql: str) -> None:
    """Validate that the query is a safe, read-only SELECT statement."""
    clean = sql.strip()
    if not clean:
        raise ValueError("Generated SQL query is empty.")
    
    # Must start with SELECT or WITH (for CTE queries)
    if not (clean.upper().startswith("SELECT") or clean.upper().startswith("WITH")):
        raise ValueError(f"Unsafe SQL detected: Only SELECT statements are permitted. Query started with: {clean[:30]}...")
        
    # Check forbidden keywords (case-insensitive whole-word match)
    for pattern in FORBIDDEN_KEYWORDS:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            raise ValueError(f"Unsafe SQL detected: Keyword '{match.group(0)}' is not allowed in read-only queries.")
            
    # Prevent query chaining / multi-statements
    statements = [s.strip() for s in clean.split(";") if s.strip()]
    if len(statements) > 1:
        raise ValueError("Unsafe SQL detected: Multiple statements in a single query are not allowed.")


def get_db():
    """Establish connection to MySQL using environment variables."""
    db_host = os.getenv("DB_HOST", "localhost").strip()
    db_user = os.getenv("DB_USER", "root").strip()
    db_password = os.getenv("DB_PASSWORD", "root").strip()
    db_name = os.getenv("DB_NAME", "atliq_tshirts").strip()
    
    try:
        db = SQLDatabase.from_uri(
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
            sample_rows_in_table_info=3
        )
        # Verify connection
        db.run("SELECT 1")
        return db
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to MySQL database '{db_name}' on host '{db_host}'. "
            f"Please verify your MySQL credentials in .env. Details: {e}"
        )


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Instantiate Google GenAI Client with validation."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip() or key.strip() in ("YOUR_GEMINI_API_KEY", "your api key here"):
        raise ValueError(
            "GEMINI_API_KEY is missing or not configured. "
            "Please add your Gemini API key to the .env file (GEMINI_API_KEY=your_key) or provide it in the UI."
        )
    return genai.Client(api_key=key.strip())


def get_gemini_model() -> str:
    """Return configured Gemini model with default fallback."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


class FewShotSQLDatabaseChain:
    """LangChain-compatible SQL Database Chain powered by Google GenAI and semantic few-shot retrieval."""
    
    def __init__(self, db: SQLDatabase, client: genai.Client, model: str = DEFAULT_GEMINI_MODEL, k: int = 2):
        self.db = db
        self.client = client
        self.model = model
        self.k = k
        self.example_selector = get_example_selector(k=k)
        
        # Track last run metadata for UI inspection
        self.last_sql: Optional[str] = None
        self.last_result: Optional[str] = None
        self.last_examples: List[Dict[str, str]] = []
        self.last_answer: Optional[str] = None

    def _call_gemini_with_fallback(self, prompt: str, temperature: float = 0.1) -> str:
        """Call Gemini model with automatic fallback to secondary models if the primary model is unavailable."""
        candidate_models = [self.model]
        for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)
                
        last_error = None
        for m in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature
                    )
                )
                if response and response.text:
                    return response.text
                return ""
            except errors.APIError as e:
                last_error = e
                # If error is not a model not found / unsupported error, re-raise directly
                if "not found" not in str(e).lower() and "unsupported" not in str(e).lower():
                    raise
                continue
            except Exception as e:
                last_error = e
                raise
                
        raise RuntimeError(f"Failed to generate response using Gemini models {candidate_models}: {last_error}")

    def run(self, query: str) -> str:
        """Executes the full pipeline: Domain Validation -> Semantic Retrieval -> SQL Generation -> SQL Safety Check -> MySQL Execution -> Answer."""
        if not query or not query.strip():
            raise ValueError("Input query cannot be empty.")
            
        question = query.strip()

        # 0. Domain validation — must pass BEFORE semantic retrieval
        validate_domain(question)

        # 1. Retrieve semantically similar few-shot examples
        selected_examples = self.example_selector.select_examples({"input": question})
        self.last_examples = selected_examples
        
        # 2. Format database schema and table info
        table_info = self.db.get_table_info()
        
        # 3. Construct few-shot examples prompt block
        examples_str = ""
        for i, ex in enumerate(selected_examples, 1):
            examples_str += f"\nExample {i}:\nQuestion: {ex.get('Question')}\nSQLQuery: {ex.get('SQLQuery')}\n"
            
        # 4. Construct SQL generation prompt
        sql_prompt = f"""You are a MySQL expert for a T-shirt inventory database.
Given an input question, create a syntactically correct MySQL query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per MySQL.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

CRITICAL RULES:
- Return ONLY the executable SQL query. Do not include explanation, markdown formatting, or preamble.
- Do NOT include any destructive commands (NO DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE).
- Only write SELECT queries.
- This database ONLY contains T-shirt inventory data (brands, colors, sizes, prices, stock, discounts).
- Do NOT generate SQL for any product type that is NOT a T-shirt (e.g. shoes, laptops, phones, jeans, bags, watches, cars, etc.).
- NEVER silently reinterpret a non-T-shirt product as a T-shirt. If the user asks about 'Adidas shoes', the word 'shoes' makes it out-of-domain — do NOT query the t_shirts table for it.
- If the question contains a non-T-shirt product, return exactly the text: OUT_OF_DOMAIN

Database Schema & Sample Rows:
{table_info}

Relevant Examples:
{examples_str}

Question: {question}
SQLQuery:"""


        # 5. Generate SQL
        raw_sql_response = self._call_gemini_with_fallback(sql_prompt, temperature=0.0)
        clean_sql = sanitize_and_extract_sql(raw_sql_response)
        self.last_sql = clean_sql
        
        # 6. Validate SQL Safety
        validate_sql(clean_sql)
        
        # 7. Execute SQL on MySQL
        try:
            sql_result = self.db.run(clean_sql)
            self.last_result = str(sql_result)
        except Exception as e:
            raise RuntimeError(f"SQL execution failed for query: '{clean_sql}'. Error: {e}")
            
        # 8. Synthesize final natural language answer
        answer_prompt = f"""You are an assistant for a T-shirt store database.
Given the user's question, the SQL query that was executed, and the SQL query result from the database, formulate a clear, helpful, and concise answer for the user.

Question: {question}
SQLQuery: {clean_sql}
SQLResult: {sql_result}

Answer:"""

        final_answer = self._call_gemini_with_fallback(answer_prompt, temperature=0.1).strip()
        self.last_answer = final_answer
        return final_answer

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain standard chain call interface."""
        query = inputs.get("query") or inputs.get("input") or ""
        result = self.run(query)
        return {"result": result, "query": query, "sql": self.last_sql}


def get_few_shot_db_chain(api_key: Optional[str] = None, model: Optional[str] = None, k: int = 2) -> FewShotSQLDatabaseChain:
    """Creates and returns the FewShotSQLDatabaseChain configured with MySQL, Google GenAI, and FAISS."""
    db = get_db()
    client = get_gemini_client(api_key=api_key)
    gemini_model = model or get_gemini_model()
    return FewShotSQLDatabaseChain(db=db, client=client, model=gemini_model, k=k)
