import os
import re
import email
from email import policy
from email.parser import BytesParser
import json
import argparse
from urllib.parse import urlparse


def extract_urls(text):
    """Extract URLs from text content using regex."""
    # This regex pattern matches URLs starting with http:// or https://
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    
    # Find all matches in the text
    matches = url_pattern.findall(text)
    
    # Return unique URLs
    return list(set(matches))


def parse_eml_file(file_path):
    """
    Parse an .eml file and extract key features:
    - Sender's email
    - Subject line
    - Email body (text)
    - URLs found in the email
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Open and parse the .eml file
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    # Extract sender's email
    from_header = msg.get('From', '')
    # Extract email address from the From header using regex
    sender_match = re.search(r'<([^>]+)>', from_header)
    if sender_match:
        sender_email = sender_match.group(1)
    else:
        # If no angle brackets, use the whole From header as email
        sender_email = from_header
    
    # Extract subject
    subject = msg.get('Subject', '')
    
    # Extract body content
    if msg.is_multipart():
        # If the message has multiple parts, try to get the text part
        body = ""
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    part_body = part.get_content()
                    body += part_body + "\n"
                except:
                    continue
    else:
        # For single part messages
        try:
            body = msg.get_content()
        except:
            body = str(msg.get_payload())
    
    # Extract URLs from the body
    urls = extract_urls(body)
    
    # Return the extracted features
    result = {
        "sender_email": sender_email.strip(),
        "subject": subject.strip(),
        "body": body.strip(),
        "urls": urls
    }
    
    return result


def convert_eml_to_text(eml_file, output_file=None):
    """
    Convert an .eml file to a regular text file with extracted features.
    """
    # Parse the .eml file
    try:
        result = parse_eml_file(eml_file)
        
        # Format the output text
        output_text = f"Sender: {result['sender_email']}\n"
        output_text += f"Subject: {result['subject']}\n\n"
        output_text += f"Body:\n{result['body']}\n\n"
        
        if result['urls']:
            output_text += "URLs:\n"
            for url in result['urls']:
                output_text += f"- {url}\n"
        
        # If output file is specified, write to it
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
        
        return output_text, result
    
    except Exception as e:
        error_msg = f"Error converting {eml_file}: {str(e)}"
        print(error_msg)
        return error_msg, None


def parse_eml_from_text(eml_content):
    """
    Parse EML content from a text string instead of a file.
    Useful for processing example emails without writing to disk.
    """
    try:
        # Convert string to bytes
        eml_bytes = eml_content.encode('utf-8')
        
        # Parse the email from bytes
        msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)
        
        # Extract sender's email
        from_header = msg.get('From', '')
        sender_match = re.search(r'<([^>]+)>', from_header)
        if sender_match:
            sender_email = sender_match.group(1)
        else:
            sender_email = from_header
        
        # Extract subject
        subject = msg.get('Subject', '')
        
        # If the message doesn't have proper headers, try to parse from the content
        if not sender_email and not subject:
            lines = eml_content.splitlines()
            if len(lines) >= 2:
                # First line might be sender
                if '<' in lines[0] and '>' in lines[0]:
                    sender_match = re.search(r'<([^>]+)>', lines[0])
                    if sender_match:
                        sender_email = sender_match.group(1)
                    else:
                        sender_email = lines[0]
                    
                    # Second line might be subject
                    subject = lines[1]
        
        # Extract body by simply joining lines after the headers
        body = eml_content
        
        # Extract URLs from the body
        urls = extract_urls(body)
        
        # Return the extracted features
        result = {
            "sender_email": sender_email.strip(),
            "subject": subject.strip(),
            "body": body.strip(),
            "urls": urls
        }
        
        # Format the output text
        output_text = f"Sender: {result['sender_email']}\n"
        output_text += f"Subject: {result['subject']}\n\n"
        output_text += f"Body:\n{result['body']}\n\n"
        
        if result['urls']:
            output_text += "URLs:\n"
            for url in result['urls']:
                output_text += f"- {url}\n"
        
        return output_text, result
        
    except Exception as e:
        error_msg = f"Error parsing email content: {str(e)}"
        print(error_msg)
        return error_msg, None


def main():
    parser = argparse.ArgumentParser(description='Convert .eml files to text files with extracted features.')
    parser.add_argument('input', help='Input .eml file or directory containing .eml files')
    parser.add_argument('-o', '--output', help='Output file or directory. If not specified, results will be printed.')
    parser.add_argument('-j', '--json', action='store_true', help='Output in JSON format instead of text')
    args = parser.parse_args()
    
    # Handle Windows-style path arguments (convert /path to regular path)
    input_path = args.input
    output_path = args.output
    
    # Remove leading slash if present (Windows command line style)
    if input_path.startswith('/'):
        input_path = input_path[1:]
    if output_path and output_path.startswith('/'):
        output_path = output_path[1:]
    
    print(f"Processing input: {input_path}")
    print(f"Output destination: {output_path}")
    
    if os.path.isdir(input_path):
        # Process all .eml files in the directory
        if output_path and not os.path.isdir(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")
        
        eml_count = 0
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.eml'):
                eml_count += 1
                input_file = os.path.join(input_path, filename)
                base_filename = os.path.splitext(filename)[0]
                
                print(f"Processing file {eml_count}: {filename}")
                
                if output_path:
                    if args.json:
                        output_file = os.path.join(output_path, f"{base_filename}.json")
                    else:
                        output_file = os.path.join(output_path, f"{base_filename}.txt")
                else:
                    output_file = None
                
                try:
                    text, result = convert_eml_to_text(input_file)
                    
                    if output_path:
                        if args.json and result:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=4)
                            print(f"  ✓ JSON result saved to {output_file}")
                        else:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(text)
                            print(f"  ✓ Text result saved to {output_file}")
                    else:
                        if args.json and result:
                            print(json.dumps(result, indent=4))
                        else:
                            print(text)
                            print("=" * 80)
                except Exception as e:
                    print(f"  ✗ Error processing {filename}: {str(e)}")
                    
        print(f"\nCompleted processing {eml_count} .eml files")
        
    else:
        # Process a single file
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} does not exist.")
            return
            
        if output_path:
            # Check if output is a directory or a file
            if os.path.isdir(output_path) or output_path.endswith('/') or output_path.endswith('\\'):
                # It's a directory
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                base_filename = os.path.basename(input_path)
                base_filename = os.path.splitext(base_filename)[0]
                
                if args.json:
                    output_file = os.path.join(output_path, f"{base_filename}.json")
                else:
                    output_file = os.path.join(output_path, f"{base_filename}.txt")
            else:
                # It's a specific file
                output_file = output_path
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
        else:
            output_file = None
        
        print(f"Processing single file: {input_path}")
        try:
            text, result = convert_eml_to_text(input_path)
            
            if output_file:
                if args.json and result:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=4)
                    print(f"✓ JSON result saved to {output_file}")
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"✓ Text result saved to {output_file}")
            else:
                if args.json and result:
                    print(json.dumps(result, indent=4))
                else:
                    print(text)
        except Exception as e:
            print(f"✗ Error processing {input_path}: {str(e)}")


if __name__ == "__main__":
    main()