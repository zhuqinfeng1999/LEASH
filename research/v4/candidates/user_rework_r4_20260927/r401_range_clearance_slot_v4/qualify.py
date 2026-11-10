"""Same unique-case launcher; only own new source/output/script binding."""
from types import FunctionType
from ..r401_normal_crib_first_risk_v3 import qualify as old
from .scripts import run_case,HERE,ROOT
ns={**old.__dict__,'run_case':run_case,'HERE':HERE,'ROOT':ROOT}
hashes=FunctionType(old.hashes.__code__,ns)
ns['hashes']=hashes
main=FunctionType(old.main.__code__,ns)
if __name__=='__main__':raise SystemExit(main())
