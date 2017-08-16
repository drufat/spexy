from spexy.sim.vorticity.vorticity import vort_save, vort_ww, vort_vv

if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        vort_save, vort_ww, vort_vv,
    ])
    parser.dispatch()
