import os

def replace_in_templates():
    template_dir = 'templates'
    old_text = 'DiabetesCare'
    new_text = 'Diabetes Detection System'
    
    for root, dirs, files in os.walk(template_dir):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if old_text in content:
                    print(f"Updating {file_path}...")
                    new_content = content.replace(old_text, new_text)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

if __name__ == '__main__':
    replace_in_templates()
