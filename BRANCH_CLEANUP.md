# Git Branch Cleanup Documentation

## Current Branch Status

As of the last check, the repository has **35+ branches** that need to be cleaned up:

### Branch Categories:
- **main** - Primary branch (KEEP)
- **copilot/** branches (31 branches) - Automated fix branches
- **codex/** branches (4 branches) - Code generation branches

## Why This Cleanup Is Needed

The user requested: "I only need main since its all merged"

All the copilot and codex branches were temporary working branches that have been merged to main. Keeping them clutters the repository and makes it harder to manage.

## How to Clean Up Branches

### Option 1: Manual Cleanup via GitHub Web Interface

1. Go to https://github.com/spotty118/Rustos/branches
2. Review each branch to confirm it's been merged
3. Click the delete button next to each branch (except main)

### Option 2: Automated Script (RECOMMENDED)

A script has been provided: `cleanup_branches.sh`

**To use the script:**

```bash
# Navigate to repository
cd /path/to/Rustos

# Run the cleanup script
./cleanup_branches.sh
```

**What the script does:**
- Lists all branches except main
- Asks for confirmation before proceeding
- Deletes all remote branches except main
- Shows remaining branches after cleanup

### Option 3: Manual Git Commands

If you prefer manual control, you can delete branches one by one:

```bash
# Delete a specific branch
git push origin --delete branch-name

# Example:
git push origin --delete copilot/fix-b1d12882-0201-4bd9-8a59-54ae644d52bc
```

### Option 4: Bulk Delete with Git

```bash
# Delete all copilot branches
git branch -r | grep "copilot/" | sed 's/origin\///' | xargs -I {} git push origin --delete {}

# Delete all codex branches  
git branch -r | grep "codex/" | sed 's/origin\///' | xargs -I {} git push origin --delete {}
```

## Important Notes

1. **Backup**: Before deleting branches, ensure all important work is merged to main
2. **Permissions**: You need write access to the repository to delete remote branches
3. **Cannot be undone**: Deleted branches cannot be easily recovered
4. **PRs**: Ensure any open Pull Requests are merged or closed before deleting their branches

## Verification After Cleanup

After running the cleanup, verify only main remains:

```bash
git branch -r
```

Expected output:
```
origin/main
```

## Alternative: GitHub CLI

If you have GitHub CLI installed:

```bash
# List all branches
gh api repos/spotty118/Rustos/branches --jq '.[].name'

# Delete branches (requires confirmation)
gh api repos/spotty118/Rustos/branches --jq '.[].name' | grep -v "main" | xargs -I {} gh api -X DELETE repos/spotty118/Rustos/git/refs/heads/{}
```

## Why This Can't Be Done Automatically

GitHub Copilot code agents have security restrictions:
- Cannot push branch deletions to remote repositories
- Cannot use force operations
- Limited to local repository modifications only

Therefore, the repository owner must perform this cleanup manually or using the provided script with their own credentials.

---

**Generated**: Automated by GitHub Copilot
**Task**: Convert simulation code and clean git branches
**Status**: Simulation code converted ✓, Branch cleanup requires manual action ⚠️
