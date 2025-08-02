import yaml

def generate_art9_prompt_template(original_domain, disguise_domain, 
                           yaml_file="prompts/formatting.yaml",
                           template_file="prompts/art9universal.txt"):
    """
    Generate a prompt template with only original_domain and disguise_domain as inputs.
    All other variables are automatically populated from the YAML file based on disguise_domain.
    """
    
    # Load the YAML template
    with open(yaml_file, 'r') as f:
        template_data = yaml.safe_load(f)
    
    # Load the prompt template from the existing file
    with open(template_file, 'r') as f:
        prompt_template = f.read()
    
    # Get the disguise domain data
    disguise_data = {
        'overall_goals': template_data['overall_goals'][disguise_domain],
        'procedural_stages': template_data['procedural_stages'][disguise_domain],
        'abstract_parameters': template_data['abstract_parameters'][disguise_domain],
        'abstract_parameter_definitions': template_data['abstract_parameter_definitions'][disguise_domain],
        'describe_actions': template_data['describe_actions'][disguise_domain],
        'new_abstract_labels': template_data['new_abstract_labels'][disguise_domain],
        'example_calculation': template_data['example_calculation'][disguise_domain]
    }
    
    # Replace all the variables in the template
    replacements = {
        '$DISGUISE_DOMAIN': disguise_domain.title(),
        '$ORIGINAL_DOMAIN': original_domain.title(),
        '$OVERALL_GOAL_EXAMPLES': disguise_data['overall_goals'],
        '$PROCEDURAL_STAGES': disguise_data['procedural_stages'],
        '$ABSTRACT_PARAMETERS': disguise_data['abstract_parameters'],
        '$ABSTRACT_PARAMETER_DEFINITIONS': disguise_data['abstract_parameter_definitions'],
        '$DESCRIBE_ACTIONS': disguise_data['describe_actions'],
        '$NEW_ABSTRACT_LABELS': disguise_data['new_abstract_labels'],
        '$EXAMPLE_CALCULATION': disguise_data['example_calculation']
    }
    
    # Apply all replacements
    for placeholder, replacement in replacements.items():
        prompt_template = prompt_template.replace(placeholder, replacement)
    
    return prompt_template

def generate_art4_prompt_template(original_domain, disguise_domain, template_file="prompts/art4universal.txt"):
    """
    Generate a prompt template with only original_domain and disguise_domain as inputs.
    """

    with open(template_file, 'r') as f:
        prompt_template = f.read()

    replacements = {
        '$ORIGINAL_DOMAIN': original_domain,
        '$DISGUISE_DOMAIN': disguise_domain
    }
    
    for placeholder, replacement in replacements.items():
        prompt_template = prompt_template.replace(placeholder, replacement)

    return prompt_template


def generate_art26_prompt_template(original_domain, disguise_domain, yaml_file="prompts/formatting.yaml", template_file="prompts/art26universal.txt"):
    """
    Generate a prompt template with only original_domain and disguise_domain as inputs.
    """
    with open(yaml_file, 'r') as f:
        template_data = yaml.safe_load(f)

    abstract_parameters = template_data['abstract_parameters'][disguise_domain]

    with open(template_file, 'r') as f:
        prompt_template = f.read()
        
    replacements = {
        '$ORIGINAL_DOMAIN': original_domain,
        '$DISGUISE_DOMAIN': disguise_domain,
        '$ABSTRACT_PARAMETERS': abstract_parameters
    }

    for placeholder, replacement in replacements.items():
        prompt_template = prompt_template.replace(placeholder, replacement)

    return prompt_template

def generate_monitor_prompt(original_domain, template_file="prompts/monitoruniversal.txt"):
    """
    Generate a monitor prompt with only original_domain as input.
    """
    # Load the prompt template from the existing file
    with open(template_file, 'r') as f:
        prompt_template = f.read()
    
    # Replace all the variables in the template
    replacements = {
        '$ORIGINAL_DOMAIN': original_domain,
    }

    for placeholder, replacement in replacements.items():
        prompt_template = prompt_template.replace(placeholder, replacement)
    
    return prompt_template
    
    

# Example usage:
if __name__ == "__main__":
    # Generate a prompt where physics problems are disguised as biology procedures
    art4_prompt = generate_art4_prompt_template("Physics", "Biology")
    with open("result.txt", "w") as f:
        f.write(art4_prompt)
