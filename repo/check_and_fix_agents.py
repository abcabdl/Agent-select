#!/usr/bin/env python3
"""Check agents and remove those without tools."""
import sqlite3
import json

DB_PATH = "demo_registry.sqlite"

def check_agents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all agents (cards with kind='agent')
    cursor.execute("SELECT id, name, role_tags, tool_tags FROM cards WHERE kind='agent'")
    agents = cursor.fetchall()
    
    print(f"Total agents: {len(agents)}\n")
    
    agents_without_tools = []
    agents_with_tools = []
    
    for agent_id, name, role_tags_json, tool_tags_json in agents:
        # Parse JSON
        role_tags = json.loads(role_tags_json) if role_tags_json else []
        tool_tags = json.loads(tool_tags_json) if tool_tags_json else []
        
        # Check if agent has tools by checking tool_tags
        if not tool_tags:
            agents_without_tools.append((agent_id, name, role_tags))
            print(f"❌ NO TOOLS: {name} (id={agent_id}, role_tags={role_tags})")
        else:
            # Check if tools actually exist
            cursor.execute("""
                SELECT COUNT(*) FROM cards 
                WHERE kind='tool'
                AND json_extract(tool_tags, '$') IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM json_each(cards.tool_tags) 
                    WHERE json_each.value IN (SELECT json_each.value FROM json_each(?))
                )
            """, (tool_tags_json,))
            tool_count = cursor.fetchone()[0]
            
            if tool_count == 0:
                agents_without_tools.append((agent_id, name, role_tags))
                print(f"❌ NO MATCHING TOOLS: {name} (id={agent_id}, role_tags={role_tags}, tool_tags={tool_tags})")
            else:
                agents_with_tools.append((agent_id, name, role_tags, tool_count))
                print(f"✅ {tool_count:3d} tools: {name} (role_tags={role_tags})")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Agents with tools: {len(agents_with_tools)}")
    print(f"  Agents without tools: {len(agents_without_tools)}")
    
    if agents_without_tools:
        print(f"\n{'='*80}")
        print("Agents to delete:")
        for agent_id, name, role_tags in agents_without_tools:
            print(f"  - {name} (id={agent_id})")
        
        response = input(f"\nDelete {len(agents_without_tools)} agents without tools? [y/N]: ")
        if response.lower() == 'y':
            for agent_id, name, role_tags in agents_without_tools:
                cursor.execute("DELETE FROM cards WHERE id = ?", (agent_id,))
                print(f"  Deleted: {name}")
            conn.commit()
            print(f"\n✓ Deleted {len(agents_without_tools)} agents")
        else:
            print("\nCancelled")
    
    conn.close()

if __name__ == "__main__":
    check_agents()
