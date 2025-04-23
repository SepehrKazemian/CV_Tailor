import logging
import os
# Import the new preprocessor and the builder
from resume_tailor.tailors.skill_graph.skill_preprocessing import SkillPreprocessor
from resume_tailor.tailors.skill_graph.graph_builder import SkillGraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Skill Graph Update Process ---")

    try:
        # Initialize components
        # Pass specific file paths here if different from defaults in SkillPreprocessor
        preprocessor = SkillPreprocessor()
        builder = SkillGraphBuilder() # Handles Neo4j connection

        # Initialize L3/L2 structure (always run, it's idempotent)
        builder.initialize_l3_domains()
        builder.initialize_l2_categories()

        # Clear skill sets for this run before preprocessing
        builder.candidate_skills.clear()
        builder.jd_skills.clear()

        # 1. Preprocess Sources (handles change detection and calls builder.process_*)
        preprocessor.preprocess_sources(builder)

        # 2. Update L1 Node Properties (Flags and Score) - Always run this
        logger.info("Updating L1 node properties based on current run...")
        builder.update_skill_properties() # Uses skills populated in builder instance this run

        # 3. Consolidate Links (check all L1 against all L2)
        logger.info("Running link consolidation...")
        builder.consolidate_all_links()

        # 4. Link Orphans (final check)
        logger.info("Running orphan linking...")
        builder.link_orphans()

        # 5. Propagate Scores (L1 -> L2 -> L3)
        logger.info("Propagating scores...")
        builder.propagate_scores()

        # 6. Final Verification & Output
        logger.info("--- Final Verification: Checking for Orphans ---")
        orphan_check_query = "MATCH (l1:L1) WHERE NOT (l1)-[:BELONGS_TO]->(:L2) RETURN l1.name as orphan_name"
        orphans = builder.connector.execute_query(orphan_check_query)
        if not orphans:
            logger.info("Verification PASSED: All L1 nodes have at least one L2 parent.")
        else:
            logger.error(f"Verification FAILED: Found {len(orphans)} orphan L1 nodes:")
            for orphan in orphans: logger.error(f"- {orphan['orphan_name']}")

        # --- Display L2s with score > 0 and their L1 children with score > 0 ---
        logger.info("--- Test: Listing L2s (>0 score) and their matched L1s (>0 score) ---")
        matched_skills_query = """
        MATCH (l1:L1)-[:BELONGS_TO]->(l2:L2)
        WHERE l2.l2_score > 0 AND l1.match_score > 0
        WITH l2, l1 ORDER BY l1.name // Order L1s alphabetically within each L2 group
        RETURN l2.name as l2_name, l2.l2_score as l2_score, collect(l1.name) as matched_l1_skills
        ORDER BY l2.l2_score DESC, l2.name // Order L2s by score desc, then name asc
        """
        matched_results = builder.connector.execute_query(matched_skills_query)

        if matched_results:
            print("\nL2 Nodes with Score > 0 and their Matched L1 Children (Score > 0):")
            for record in matched_results:
                print(f"\n  L2: {record['l2_name']} (Score: {record['l2_score']})")
                if record['matched_l1_skills']:
                    for skill in record['matched_l1_skills']:
                        print(f"    - L1: {skill}")
                else:
                     print("     (No L1 children with score > 0 found)")
        else:
            print("\nNo L2 nodes found with score > 0 having L1 children with score > 0.")


        # Close connection
        builder.close_connection()
        logger.info("--- Skill Graph Update Process Finished ---")

    except ConnectionError as e:
        logger.error(f"Process failed due to Neo4j connection error: {e}")
    except FileNotFoundError as e:
         logger.error(f"Process failed due to missing file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the graph update process: {e}", exc_info=True)
