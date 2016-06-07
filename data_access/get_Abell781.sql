# This is for querying galaxies in F5 of the Deep Lens Survey
# Comment out the last line if you wish to query the entire field

SELECT
p.alpha,
p.delta,
# p.r,
s.e1,
s.e2,
s.de
# s.a,
# s.b,
# p.processflags,
# z.z_b,
# s.flux_radius,
# d.Dlsqc_prob,
# There is special code in query_DLS.py to handle column names for this wild card entry
# r.*
FROM
RC1Stage.PhotoObjAll AS p,
RC1c_public.Dlsqc AS d,
RC1c_public.Bpz AS z,
RC1Stage.Shapes2 AS s,
RC1c_public.Probs AS r
WHERE
d.objid=s.objid
AND p.objid = s.objid
AND p.objid = z.objid
AND p.objid = r.objid
AND p.objid IS NOT NULL
AND p.processflags<8
AND p.r is NOT NULL
AND p.r>21 AND p.r<27
# The R band probability that object is a point source `d.Dlsqc_prob`
AND d.Dlsqc_prob<0.1
# Shape cut
AND s.b>0.4
AND z.z_b>0.5
# Ellipticity error cut
AND s.de<0.1
# F5 bound cut F5 is a 2x2 sq. degree field centered at RA=13:59:20, DEC=-11:03:00
# Due to ambiguous info on DLS website and James 2015 cosmic shear paper about the location
# of the field I will use the SQL keyword for specifying subfield instead
AND p.alpha between 139.8 and 140.403
AND p.delta between 30.2056 and 30.8056
AND p.subfield LIKE 'F2%';
# AND p.subfield = r.subfield
# LIMIT 100;
