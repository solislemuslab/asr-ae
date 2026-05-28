#!/bin/bash

# Check if the first positional argument is provided
if [ -z "$1" ]; then
  echo "Error: No command provided. Use 'train', 'decimate' or 'sample'."
  exit 1
fi

# Assign the first positional argument to a variable
COMMAND=$1
shift # Remove the first positional argument, so "$@" now contains only the optional arguments

# Map the command to the corresponding script
case "$COMMAND" in
  train)
    if [ -z "$1" ]; then
        echo "Error: No sub-command provided for 'train'. Use 'bmDCA', 'eaDCA' or 'edDCA'."
        exit 1
    fi
    
    SUBCOMMAND=$1
    shift
    case "$SUBCOMMAND" in
      -m) 
        SUBCOMMAND=$1
        shift
        case "$SUBCOMMAND" in
          bmDCA)
            exec="bmDCA"
            SUBCOMMAND=$1
            shift
          ;;
          eaDCA)
            exec="eaDCA"
            SUBCOMMAND=$1
            shift
          ;;
          edDCA)
            exec="edDCA"
            SUBCOMMAND=$1
            shift
          ;;
          *)
            echo "1 Error: Invalid sub-command '$SUBCOMMAND' for 'train'. Use -m followed by 'bmDCA', 'eaDCA' or 'edDCA'."
            exit 1
          ;;
        esac
        ;;
      *)
        exec="bmDCA"
        # echo "Error: Invalid sub-command '$SUBCOMMAND' for 'train'. Use -m followed by 'bmDCA', 'eaDCA' or 'edDCA'."
        # exit 1
          ;;
      esac
    ;;
  sample)
    exec="sample"
    ;;
  importance_sample)
    exec="importance_sample"
    ;;
  TD_integration)
    exec="TD_integration"
    ;;
  energies)
    exec="energies"
    ;;
  DMS)
    exec="DMS"
    ;;
  contacts)
    exec="contacts"
    ;;
  *)
    echo "Error: Invalid command '$COMMAND'. Use 'train', 'decimate', 'sample', 'energies', 'DMS' or 'contacts'."

    exit 1
    ;;
esac

# Parse --nthreads from remaining arguments (default to 4)
NTHREADS=4
all_args=("$SUBCOMMAND" "$@")
for i in "${!all_args[@]}"; do
  if [ "${all_args[$i]}" = "--nthreads" ]; then
    NTHREADS="${all_args[$((i+1))]}"
    break
  fi
done
export JULIA_NUM_THREADS=$NTHREADS

# Run the corresponding Julia script with the remaining optional arguments
julia --project=. scripts/adabmDCA/execute.jl -m "$exec" "$SUBCOMMAND""$@" & disown










