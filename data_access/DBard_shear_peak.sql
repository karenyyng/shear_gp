# This is a modified SQL query from the Bard, D. 2014 shear peak paper
# > mysql -h matilda.physics.ucdavis.edu -u DLS -p
# Ask Debbie Bard <djbard@lbl.gov> for password 
# > use RC1Stage <- database name 

SELECT  
p.alpha, 
p.delta, 
p.r, 
s.e1, 
s.e2, 
s.de, 
s.a, 
s.b, 
p.processflags, 
z.z_b, 
s.flux_radius, 
d.Dlsqc_prob 
FROM 
RC1Stage.PhotoObjAll AS p, 
RC1c_public.Dlsqc AS d, 
RC1c_public.Bpz AS z, 
RC1Stage.Shapes2 AS s 
WHERE 
d.objid=s.objid 
AND p.objid = s.objid 
AND p.objid = z.objid  
AND p.objid IS NOT NULL 
AND p.processflags<8  
AND p.r is NOT NULL 
AND p.r>21 AND p.r<27  
# The R band probability that object is a point source `d.Dlsqc_prob`
AND d.Dlsqc_prob<0.1 
# Shape cut 
AND s.b>0.4 
AND z.z_b>0.3 
# Ellipticity error cut 
AND s.de<0.25 
# F5 bound cut F5 is a 2x2 sq. degree field centered at RA=13:59:20, DEC=-11:03:00
AND p.alpha between 208.7 and 210.85
AND p.delta between -12.1 and -10.1
limit 2000;
