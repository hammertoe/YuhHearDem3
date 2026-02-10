#!/bin/bash

# Database management script for Parliamentary Search System

set -e

DB_NAME="parliament_search"
DB_USER="postgres"
DB_HOST="localhost"
DB_PORT="5432"
DB_PASSWORD="postgres"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Parliamentary Search Database Manager ===${NC}"

# Function to check if container is running
check_containers() {
    echo -e "\n${YELLOW}Checking containers...${NC}"
    
    if docker ps | grep -q "parliament_postgres"; then
        echo -e "${GREEN}✓${NC} PostgreSQL container running"
    else
        echo -e "${RED}✗${NC} PostgreSQL container NOT running"
        echo "Run: docker-compose up -d"
        exit 1
    fi
    
    if docker ps | grep -q "parliament_memgraph"; then
        echo -e "${GREEN}✓${NC} Memgraph container running"
    else
        echo -e "${RED}✗${NC} Memgraph container NOT running"
        echo "Run: docker-compose up -d"
        exit 1
    fi
}

# Function to wait for database to be ready
wait_for_db() {
    echo -e "\n${YELLOW}Waiting for database to be ready...${NC}"
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec parliament_postgres pg_isready -U postgres > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Database is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "\n${RED}✗${NC} Database did not become ready in 60 seconds"
    echo "Check logs: docker logs parliament_postgres"
    exit 1
}

# Function to connect to database
connect_db() {
    echo -e "\n${YELLOW}Connecting to database...${NC}"
    docker exec -it parliament_postgres psql -U $DB_USER -d $DB_NAME
}

# Function to check schema
check_schema() {
    echo -e "\n${YELLOW}Checking database schema...${NC}"
    docker exec parliament_postgres psql -U $DB_USER -d $DB_NAME -c "\dt"
}

# Function to check tables exist
check_tables() {
    echo -e "\n${YELLOW}Checking tables...${NC}"
    docker exec parliament_postgres psql -U $DB_USER -d $DB_NAME -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
}

# Function to show database info
db_info() {
    echo -e "\n${BLUE}=== Database Information ===${NC}"
    docker exec parliament_postgres psql -U $DB_USER -d $DB_NAME -c "
        SELECT 
            schemaname,
            tablename,
            tableowner
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename;
    "
}

# Function to create test data
create_test_data() {
    echo -e "\n${YELLOW}Creating test data...${NC}"
    docker exec parliament_postgres psql -U $DB_USER -d $DB_NAME -c "
        -- Insert test speaker
        INSERT INTO speakers (id, normalized_name, full_name, title, position, role_in_video, first_appearance, total_appearances)
        VALUES ('s_test_1', 'test', 'Test Speaker', 'Hon.', 'Member', 'member', '00:00:00', 1)
        ON CONFLICT (id) DO NOTHING;
        
        -- Check it worked
        SELECT * FROM speakers LIMIT 5;
    "
}

# Main menu
show_menu() {
    echo -e "\n${BLUE}Select an option:${NC}"
    echo "1. Check containers status"
    echo "2. Wait for database to be ready"
    echo "3. Connect to database"
    echo "4. Check schema"
    echo "5. Check tables"
    echo "6. Show database information"
    echo "7. Create test data"
    echo "8. View container logs (PostgreSQL)"
    echo "9. View container logs (Memgraph)"
    echo "0. Exit"
    echo -n -e "\n${YELLOW}Enter choice [0-9]: ${NC}"
    read choice
}

# Parse command line arguments
if [ $# -gt 0 ]; then
    case $1 in
        check)
            check_containers
            ;;
        wait)
            wait_for_db
            ;;
        connect)
            check_containers
            wait_for_db
            connect_db
            ;;
        schema)
            check_containers
            wait_for_db
            check_schema
            ;;
        tables)
            check_containers
            wait_for_db
            check_tables
            ;;
        info)
            check_containers
            wait_for_db
            db_info
            ;;
        test)
            check_containers
            wait_for_db
            create_test_data
            ;;
        logs-postgres)
            docker logs parliament_postgres --tail 100
            ;;
        logs-memgraph)
            docker logs parliament_memgraph --tail 100
            ;;
        *)
            echo "Unknown command: $1"
            echo "Usage: $0 {check|wait|connect|schema|tables|info|test|logs-postgres|logs-memgraph}"
            exit 1
            ;;
    esac
else
    # Interactive mode
    while true; do
        show_menu
        case $choice in
            1)
                check_containers
                ;;
            2)
                wait_for_db
                ;;
            3)
                check_containers
                wait_for_db
                connect_db
                ;;
            4)
                check_containers
                wait_for_db
                check_schema
                ;;
            5)
                check_containers
                wait_for_db
                check_tables
                ;;
            6)
                check_containers
                wait_for_db
                db_info
                ;;
            7)
                check_containers
                wait_for_db
                create_test_data
                ;;
            8)
                docker logs parliament_postgres --tail 100
                ;;
            9)
                docker logs parliament_memgraph --tail 100
                ;;
            0)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                ;;
        esac
    done
fi
