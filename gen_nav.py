
import os
import yaml
basic_dir = "docs/"

# 遍历文件夹并构造nav部分
def generate_nav(directory):
    queue = [directory]
    res   = []
    while len(queue) > 0:
        cur_dir = queue.pop(0)
        cur_res = []
        for root, dirs, files in os.walk(cur_dir):
            for file in files:
                if file.endswith('.md'):
                    cur_res.append(os.path.join(root, file).replace(basic_dir, ''))
            break
        cur_res.sort()
        if len(cur_res) > 0:
            res.append({cur_dir.replace(basic_dir, '') : cur_res})
        for dir in dirs:
            queue.append(os.path.join(cur_dir, dir))
    return res

# 更新mkdocs.yml
def update_mkdocs_yml(nav):
    with open('mkdocs.yml', 'r') as f:
        config = yaml.safe_load(f)

    config['nav'] = nav

    with open('mkdocs.yml', 'w') as f:
        yaml.dump(config, f)

nav = generate_nav('docs')
print(nav)
update_mkdocs_yml(nav)