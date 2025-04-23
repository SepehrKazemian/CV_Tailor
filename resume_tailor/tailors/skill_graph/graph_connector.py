import logging
import importlib.util
from pathlib import Path
from neo4j import GraphDatabase, Driver, Session, Transaction
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# --- Load Credentials Safely ---
try:
    # Assumes credentials.py is in the root directory, 3 levels up from skill_graph
    creds_path = Path(__file__).resolve().parents[3] / "credentials.py"
    spec_creds = importlib.util.spec_from_file_location("credentials", creds_path)
    credentials = importlib.util.module_from_spec(spec_creds)
    spec_creds.loader.exec_module(credentials)
    NEO4J_URI = getattr(credentials, "NEO4J_URI", None)
    NEO4J_USER = getattr(credentials, "NEO4J_USER", None)
    NEO4J_PASSWORD = getattr(credentials, "NEO4J_PASSWORD", None)
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        logger.error("Neo4j credentials (URI, USER, PASSWORD) not found in credentials.py.")
        # Set to None to prevent connection attempts
        NEO4J_URI = NEO4J_USER = NEO4J_PASSWORD = None
except ImportError:
    logger.error("Could not import credentials.py. Ensure it exists in the project root.")
    NEO4J_URI = NEO4J_USER = NEO4J_PASSWORD = None
except Exception as e:
    logger.error(f"Error loading credentials: {e}", exc_info=True)
    NEO4J_URI = NEO4J_USER = NEO4J_PASSWORD = None

# --- Neo4j Connection Class ---
class Neo4jConnector:
    """Handles connection and query execution for Neo4j."""

    _driver: Optional[Driver] = None

    def __init__(self):
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
            logger.error("Neo4j credentials missing. Cannot initialize connector.")
            self._driver = None
            return

        try:
            self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}", exc_info=True)
            self._driver = None

    def close(self):
        """Closes the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a read or write query against the Neo4j database.

        Args:
            query: The Cypher query string.
            parameters: Optional dictionary of parameters for the query.

        Returns:
            A list of result records (dictionaries), or an empty list if error/no results.
        """
        if not self._driver:
            logger.error("Neo4j driver not initialized. Cannot execute query.")
            return []

        records = []
        try:
            # Use session context manager for automatic resource management
            with self._driver.session() as session:
                # Determine if it's a read or write transaction based on keywords
                # This is a basic heuristic, might need refinement for complex cases
                is_write = any(kw in query.upper() for kw in ["CREATE", "MERGE", "SET", "DELETE", "REMOVE"])

                if is_write:
                    # Pass the query and parameters to execute_write, which calls _run_transaction
                    # _run_transaction now returns the processed list of records
                    records = session.execute_write(self._run_transaction, query, parameters)
                else:
                    # Pass the query and parameters to execute_read
                    records = session.execute_read(self._run_transaction, query, parameters)

                # Ensure records is always a list, even if the transaction function returns None on error
                records = records if records is not None else []
                logger.debug(f"Query executed successfully. Query: {query[:100]}..., Params: {parameters}, Results: {len(records)}")

        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}", exc_info=True)
            logger.error(f"Failed Query: {query}")
            logger.error(f"Parameters: {parameters}")
            # Optionally re-raise or handle specific Neo4j exceptions
            return [] # Return empty list on error

        return records

    @staticmethod
    def _run_transaction(tx: Transaction, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Helper function to run query within a transaction and consume results.
        Returns a list of result dictionaries.
        """
        result = tx.run(query, parameters)
        # Consume the result within the transaction by converting records to dictionaries
        try:
            records = [record.data() for record in result]
            return records
        except Exception as e:
             # Log error specific to result processing within transaction
             logger.error(f"Error processing results within transaction: {e}", exc_info=True)
             logger.error(f"Query during error: {query}")
             logger.error(f"Parameters during error: {parameters}")
             return [] # Return empty list if processing fails

# --- Singleton Instance ---
# Optional: Provide a singleton instance for easy access throughout the module
# Be mindful of connection lifecycle if using a long-lived singleton.
# graph_db_connector = Neo4jConnector()

# Example Usage (for testing or direct use):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = Neo4jConnector()
    if connector._driver:
        try:
            # Test query
            results = connector.execute_query("MATCH (n) RETURN count(n) as node_count")
            if results:
                print(f"Test query successful. Node count: {results[0]['node_count']}")
            else:
                print("Test query executed, but no results returned (or error occurred). Check logs.")

            # Example write query (MERGE ensures idempotency)
            write_results = connector.execute_query(
                "MERGE (t:TestNode {name: $name}) RETURN t.name",
                parameters={"name": "Test1"}
            )
            if write_results:
                 print(f"Test write query successful. Created/Matched node: {write_results[0]['t.name']}")

        finally:
            connector.close()
    else:
        print("Could not establish Neo4j connection. Check credentials and server status.")
