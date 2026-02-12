import sys, os, traceback
import gmsh

def main():
    if len(sys.argv) != 3:
        print("Usage: gmsh_worker.py input.geo output.msh", file=sys.stderr)
        sys.exit(2)

    geo_in, msh_out = sys.argv[1], sys.argv[2]

    try:
        # High verbosity + fail-fast; log to a child file for post-mortem
        gmsh.initialize(['-v', '1', '-log', 'gmsh.log'])
        try:
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.AbortOnError", 1)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)

            if not os.path.isfile(geo_in):
                print(f"ERROR: .geo not found: {geo_in}", file=sys.stderr)
                sys.exit(3)

            # Remove stale output
            try:
                if os.path.exists(msh_out):
                    os.remove(msh_out)
            except Exception:
                pass

            gmsh.open(geo_in)
            gmsh.model.mesh.generate(2)

            types, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
            total_elems = sum(len(tags) for tags in elemTags)
            if total_elems == 0:
                print("ERROR: Empty mesh (0 elements)", file=sys.stderr)
                sys.exit(4)

            gmsh.write(msh_out)
            if (not os.path.isfile(msh_out)) or os.path.getsize(msh_out) == 0:
                print("ERROR: .msh file missing or empty", file=sys.stderr)
                sys.exit(5)

            sys.exit(0)

        finally:
            try:
                gmsh.finalize()
            except Exception:
                pass

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
