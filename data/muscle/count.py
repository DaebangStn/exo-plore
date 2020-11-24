# Counting number of muscles which include certain body parts

import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse('data/muscle/muscle_jn_foot3.xml')  # Replace with the path to your XML file
root = tree.getroot()

# Option 1: Include specific body parts (e.g., only 'Pelvis')
lower = [
    # "Pelvis",
    "FemurL", "FemurR", "TibiaL", "TibiaR", "TalusL", "TalusL", "FootThumbL", "FootThumbR", "FootPinkyR", "FootPinkyL"]
arms = ["ArmL", "ArmR", "ForeArmL", "ForeArmR", "ShoulderR", "ShoulderL", "HandR", "HandL"]

lower = set(lower)
arms = set(arms)

total = 0
lower_muscle = []
arm_muscle = []
exclude_muscle = []

# Iterate over each <Unit> element
for unit in root.findall(".//Unit"):
    # Find all <Waypoint> elements in the current <Unit>
    waypoints = unit.findall("Waypoint")

    # Extract the 'body' attribute for all waypoints in this unit
    bodies_in_unit = {waypoint.get('body') for waypoint in waypoints}

    total += 1

    if arms.intersection(bodies_in_unit):
        arm_muscle.append(unit.get('name'))

    if lower.intersection(bodies_in_unit):
        lower_muscle.append(unit.get('name'))

    if not arms.intersection(bodies_in_unit) and not lower.intersection(bodies_in_unit):
        exclude_muscle.append(unit.get('name'))


# Print the final count
print(f"Total elements: {total}")
print(f"Lower body parts: {len(lower_muscle)}")
print(lower_muscle)
print(f"Arm parts: {len(arm_muscle)}")
print(arm_muscle)
print(f"Excluded: {len(exclude_muscle)}")
print(exclude_muscle)

