# Beyond V1

## SLAM camera-ray reflections

Eine spätere SLAM-Erweiterung kann Reflexionen mit Kamerastrahlen statt eines
Scanursprungs prüfen. Sie braucht reale NavVis-/XGRIDS-Referenzdaten und eine
separate Genauigkeitsvalidierung; Klasse 68 bleibt bis dahin im SLAM-Modus aus.

E57 classification export is intentionally excluded. A namespace-qualified
extension point field could encode class codes, but a schema, extension URI,
legend representation and importer support must be established first. RiSCAN PRO
and other target importers require explicit round-trip tests; support is UNCLEAR.
