import json
import os
import re
from typing import Any, Dict, List, Optional


class JsonDataProcessor:
    """
    A class to process and preprocess JSON files containing archived Publico website data.
    Handles either a single file or all JSON files in a directory.
    """

    def __init__(self, input_path: str, output_dir: Optional[str] = None):
        """
        Initialize the processor with input and output paths.

        Args:
            input_path (str): Path to either a JSON file or a directory containing JSON files
            output_dir (str, optional): Directory to save output files. If None,
                                        uses the same directory as the input.
        """
        self.input_path = input_path

        # Set default output directory if not provided
        if output_dir is None:
            if os.path.isdir(input_path):
                self.output_dir = input_path
            else:
                self.output_dir = os.path.dirname(input_path)
        else:
            self.output_dir = output_dir

            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.files_to_process = []
        self._identify_files_to_process()

        self.current_file = None
        self.data = None
        self.processed_data = []

    def _identify_files_to_process(self) -> None:
        """
        Identify which JSON files to process based on the input path.
        """
        if os.path.isdir(self.input_path):
            # Process all JSON files in the directory
            for filename in os.listdir(self.input_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.input_path, filename)
                    self.files_to_process.append(file_path)
            print(
                f"Found {len(self.files_to_process)} JSON files to process in {self.input_path}"
            )
        elif os.path.isfile(self.input_path) and self.input_path.endswith(".json"):
            # Process a single JSON file
            self.files_to_process.append(self.input_path)
            print(f"Will process single file: {self.input_path}")
        else:
            print(f"Error: {self.input_path} is not a valid JSON file or directory")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing HTML tags, normalizing line breaks,
        and cleaning up whitespace.

        Args:
            text (str): Raw text from the JSON file

        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""

        # Remove HTML tags if present
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove repeated empty lines (more than 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]

        # Remove empty lines at the beginning and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        # Rejoin the lines
        processed_text = "\n".join(lines)

        return processed_text

    def load_data(self, file_path: str) -> None:
        """
        Load the JSON data from the specified file.

        Args:
            file_path (str): Path to the JSON file to load
        """
        self.current_file = file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            print(f"Successfully loaded data from {file_path}")
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
            self.data = {}

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single JSON file into the desired output structure.

        Args:
            file_path (str): Path to the JSON file to process

        Returns:
            List[Dict[str, Any]]: List of processed entries
        """
        self.load_data(file_path)
        self.processed_data = []

        # Process each timestamp entry in the JSON
        for timestamp, parent_info in self.data.items():
            # Extract parent information
            parent_id = timestamp
            parent_title = parent_info.get("title", "")
            parent_originalURL = parent_info.get("originalURL", "")
            parent_linkToArchive = parent_info.get("linkToArchive", "")
            parent_linkToNoFrame = parent_info.get("linkToNoFrame", "")
            parent_tstamp = parent_info.get("tstamp", "")
            parent_linkToScreenshot = parent_info.get("linkToScreenshot", "")

            # Process children if they exist
            if "children" in parent_info and parent_info["children"]:
                for i, child in enumerate(parent_info["children"]):
                    # Preprocess the text
                    raw_text = child.get("text", "")
                    processed_text = self.preprocess_text(raw_text)

                    # Create an entry for each child
                    entry = {
                        "source": f"{os.path.basename(file_path)}/{parent_id}",
                        "link": child.get("link", ""),
                        "text": processed_text,
                        "child_id": i,
                        "parent_id": parent_id,
                        "parent_title": parent_title,
                        "parent_originalURL": parent_originalURL,
                        "parent_linkToArchive": parent_linkToArchive,
                        "parent_linkToNoFrame": parent_linkToNoFrame,
                        "parent_tstamp": parent_tstamp,
                        "parent_linkToScreenshot": parent_linkToScreenshot,
                    }
                    self.processed_data.append(entry)

        return self.processed_data

    def save_processed_data(self, file_path: str) -> None:
        """
        Save the processed data to an output file.

        Args:
            file_path (str): Path to the input file that was processed
        """
        if not self.processed_data:
            print(f"No processed data to save for {file_path}")
            return

        # Create output filename based on input filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(self.output_dir, f"processed_{base_name}.json")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
            print(f"Processed {len(self.processed_data)} entries from {file_path}")
            print(f"Output saved to {output_file}")
        except Exception as e:
            print(f"Error saving data to {output_file}: {str(e)}")

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run the complete process for all identified files.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping file paths to their processed data
        """
        results = {}

        if not self.files_to_process:
            print("No files to process")
            return results

        for file_path in self.files_to_process:
            print(f"\nProcessing {file_path}...")
            processed_data = self.process_file(file_path)
            self.save_processed_data(file_path)
            results[file_path] = processed_data

        print(f"\nFinished processing {len(self.files_to_process)} files")
        return results


def main():
    """
    Main function to run the processor from command line.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process JSON files.")
    parser.add_argument(
        "input_path", help="Path to a JSON file or directory containing JSON files"
    )
    parser.add_argument("--output-dir", "-o", help="Directory to save output files")

    args = parser.parse_args()

    # Create and run the processor
    processor = JsonDataProcessor(args.input_path, args.output_dir)
    processor.run()


if __name__ == "__main__":
    main()
