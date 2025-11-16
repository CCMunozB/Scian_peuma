#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Simple Auto-VENV Setup Script for Raspberry Pi
# This will configure your .bashrc to auto-activate venv on login

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Auto-VENV Setup Script${NC}"
echo "=============================="

# Get project directory from user
read -p "Enter your project directory path [default: /home/pi/myproject]: " PROJECT_DIR
PROJECT_DIR=${PROJECT_DIR:-/home/pi/myproject}

# Get venv name from user
read -p "Enter your venv directory name [default: venv]: " VENV_NAME
VENV_NAME=${VENV_NAME:-venv}

VENV_PATH="$PROJECT_DIR/$VENV_NAME"

echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "Project directory: $PROJECT_DIR"
echo "Virtual environment: $VENV_PATH"
echo ""

# Check if paths exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}Warning: Project directory doesn't exist. Creating it...${NC}"
    mkdir -p "$PROJECT_DIR"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Warning: Virtual environment doesn't exist at $VENV_PATH${NC}"
    read -p "Do you want to create it? (y/n): " CREATE_VENV
    if [ "$CREATE_VENV" = "y" ] || [ "$CREATE_VENV" = "Y" ]; then
        python3 -m venv "$VENV_PATH"
        echo -e "${GREEN}Virtual environment created!${NC}"
    fi
fi

# Backup existing .bashrc
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo -e "${GREEN}Backed up existing .bashrc${NC}"
fi

# Add auto-venv configuration to .bashrc
echo "" >> ~/.bashrc
echo "# Auto-activate Python Virtual Environment" >> ~/.bashrc
echo "AUTO_VENV_DIR=\"$PROJECT_DIR\"" >> ~/.bashrc
echo "AUTO_VENV_PATH=\"$VENV_PATH\"" >> ~/.bashrc
echo "if [ -f \"\$AUTO_VENV_PATH/bin/activate\" ]; then" >> ~/.bashrc
echo "    source \"\$AUTO_VENV_PATH/bin/activate\"" >> ~/.bashrc
echo "    cd \"\$AUTO_VENV_DIR\"" >> ~/.bashrc
echo "    echo -e \"${GREEN}✓ Virtual environment activated!${NC}\"" >> ~/.bashrc
echo "    echo -e \"${YELLOW}Project directory: \$AUTO_VENV_DIR${NC}\"" >> ~/.bashrc
echo "    echo \"$(python3 --version)\"" >> ~/.bashrc
echo "else" >> ~/.bashrc
echo "    echo -e \"${RED}⚠ Virtual environment not found at: \$AUTO_VENV_PATH${NC}\"" >> ~/.bashrc
echo "fi" >> ~/.bashrc

echo ""
echo -e "${GREEN}✓ Auto-VENV configuration added to .bashrc!${NC}"
echo ""
echo -e "${YELLOW}To test immediately, run:${NC}"
echo "source ~/.bashrc"
echo ""
echo -e "${YELLOW}The virtual environment will auto-activate on every login.${NC}"
echo -e "${YELLOW}To disable, remove the auto-venv section from ~/.bashrc${NC}"