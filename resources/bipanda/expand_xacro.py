import xacro
doc = xacro.process_file('bipanda_spherized_1.xacro')
xml = doc.toprettyxml()
open('bipanda_spherized_1.urdf', 'w').write(xml)

doc = xacro.process_file('bipanda_spherized.xacro')
xml = doc.toprettyxml()
open('bipanda_spherized.urdf', 'w').write(xml)