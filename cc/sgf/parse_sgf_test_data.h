#ifndef __SGF_PARSE_SGF_DATA_H_
#define __SGF_PARSE_SGF_DATA_H_

namespace sgf {

static constexpr char kSgfFF4[] = R"((;
FF[4]
EV[Kiwang, 15th]
RO[League]
PB[Ha Chan Seok]
BR[8p]
PW[Seo Bong Soo]
WR[9p]
KM[5.5]
RE[W+1.5]
DT[1990-09-11]
SZ[19]
CA[UTF-8]
RU[Korean]

;B[pd];W[dp];B[pq];W[dc];B[de];W[ce];B[cf];W[cd];B[fq];W[cn]
;B[jp];W[qn];B[po];W[rp];B[pn];W[qm];B[pl];W[ql];B[pk];W[pm]
;B[om];W[rj];B[fo];W[lq];B[kq];W[lo];B[nn];W[kn];B[in];W[kl]
;B[dg];W[eq];B[fr];W[fc];B[ck];W[er];B[ep];W[cq];B[lm];W[km]
;B[il];W[kj];B[lk];W[kk];B[ij];W[kh];B[ge];W[eo];B[fp];W[ii]
;B[hi];W[ih];B[hh];W[ig];B[nc];W[gf];B[fe];W[hg];B[fh];W[qc]
;B[qd];W[pc];B[oc];W[rd];B[re];W[rc];B[qf];W[pb];B[be];W[dl]
;B[cl];W[dm];B[bd];W[bc];B[hc];W[gc];B[hb];W[dk];B[cj];W[kd]
;B[kb];W[fs];B[gs];W[es];B[hr];W[qq];B[qr];W[dj];B[fa];W[eb]
;B[bg];W[hd];B[id];W[he];B[kc];W[ic];B[jc];W[ib];B[ia];W[ie]
;B[jd];W[gb];B[jb];W[ob];B[nb];W[rr];B[qp];W[rq];B[lr];W[mq]
;B[mr];W[kp];B[nq];W[jo];B[jq];W[io];B[qo];W[ro];B[bn];W[bo]
;B[bm];W[ke];B[ri];W[qi];B[rh];W[qj];B[mj];W[hj];B[hk];W[gj]
;B[gk];W[fj];B[ho];W[di];B[ff];W[fg];B[eg];W[gg];B[pr];W[ga]
;B[en];W[fm];B[do];W[co];B[gm];W[ac];B[dd];W[cc];B[ao];W[ap]
;B[an];W[bp];B[ll];W[mo];B[qh];W[ip];B[sj];W[rk];B[fk];W[ch]
;B[bh];W[ci];B[bi];W[fl];B[fi];W[jj];B[ik];W[ej];B[eh];W[cm]
;B[bl];W[no];B[oi];W[nf];B[je];W[kf];B[oa];W[ra];B[mh];W[mg]
;B[lh];W[lg];B[md];W[op];B[oq];W[mn];B[mm];W[ld];B[lc];W[jf]
;B[rs];W[ed];B[ee];W[nh];B[ni];W[li];B[mi];W[pj];B[oj];W[pp]
;B[on];W[sk];B[me];W[iq];B[ir];W[gp];B[go];W[gq];B[hp];W[hq]
;B[jn];W[gl];B[hm];W[jm];B[im];W[ko];B[gr];W[ae];B[og];W[ng]
;B[of];W[ne];B[si];W[oe];B[pe];W[ad];B[bf];W[fn];B[lj];W[ki]
;B[mp];W[lp];B[np];W[oo];B[ol];W[gn];B[hn];W[pa];B[na];W[af]
;B[sd];W[sc];B[se];W[ag];B[cg];W[ah];B[ai];W[ph];B[oh];W[pi]
;B[pg];W[qk];B[sr];W[sq];B[nd];W[fd];B[lf];W[em];B[qs];W[ek]
;B[ss])
)";

static constexpr char kSgfFF3[] = R"((;
FF[3]
EV[2009 Korean Baduk league (Team T Broad vs Team Batoo)]
PB[Baek Hongsuk]
BR[7d]
PW[Cho Hanseung]
WR[9d]
KM[6.5]
RE[B+R]
DT[2009-08-16]
SZ[19]

;B[pd];W[dd];B[pq];W[dp];B[qk];W[fq];B[nc];W[fc];B[cj];W[cl]
;B[cf];W[ce];B[df];W[ch];B[bh];W[dh];B[bg];W[dj];B[ed];W[ec]
;B[cc];W[pp];B[cd];W[oq];B[qq];W[qp];B[or];W[nq];B[nr];W[mq]
;B[hc];W[dc];B[de];W[qi];B[ok];W[qf];B[qe];W[pf];B[oi];W[ne]
;B[rf];W[rg];B[re];W[nh];B[ni];W[mh];B[lj];W[oh];B[md];W[pi]
;B[nn];W[rp];B[mr];W[lq];B[lr];W[kq];B[kr];W[nk];B[ol];W[oj]
;B[nl];W[mk];B[ll];W[lk];B[kk];W[kl];B[km];W[jl];B[ml];W[jk]
;B[jm];W[kj];B[jq];W[jp];B[kp];W[iq];B[jr];W[ko];B[lp];W[lo]
;B[mp];W[mo];B[np];W[no];B[op];W[oo];B[on];W[im];B[ip];W[jn]
;B[jo];W[lm];B[qn];W[po];B[rj];W[cb];B[bb];W[bc];B[bd];W[ab]
;B[ba];W[ad];B[ac];W[rl];B[rm];W[bc];B[be];W[ca];B[ac];W[ql]
;B[pl];W[bc];B[bn];W[aa];B[cq];W[cp];B[bp];W[bo];B[co];W[bq]
;B[ao];W[dq];B[do];W[bj];B[jd];W[dm];B[fo];W[cr];B[gm];W[jf]
;B[gb];W[ho];B[en];W[hp];B[em];W[ri];B[rk];W[bm];B[fj];W[fh]
;B[el];W[bi];B[cn];W[fb];B[ke];W[ir];B[kf];W[jg];B[is];W[me]
;B[le];W[gk];B[cm];W[dl];B[ej];W[al];B[dk];W[ck];B[di];W[ci]
;B[eh];W[dj];B[ei];W[ek];B[fk];W[gj];B[eg];W[dk];B[gi];W[hi]
;B[hh];W[gh];B[fi];W[hg];B[hj];W[ih];B[hk];W[ib];B[ic];W[oc]
;B[od];W[nd];B[mc];W[pc];B[qc];W[qb];B[rc];W[nb];B[mb];W[rb]
;B[oa];W[pb];B[sb];W[pe];B[qd];W[sf];B[ob];W[sd];B[sc];W[hb]
;B[gc];W[ga];B[ha])
)";

static constexpr char kSgfHandicap[] = R"((;
PB[Wada Ikkei]
BR[2d]
PW[Honinbo Shusaku]
WR[6d]
RE[W+R]
JD[Kaei 3-5-27]
DT[1850-07-06]
HA[2]
AB[dp][pd]

;W[cd];B[qo];W[op];B[pq];W[oq];B[pr];W[kq];B[ec];W[fq];B[cf]
;W[de];B[df];W[fe];B[ck];W[fc];B[fb];W[gc];B[cc];W[bc];B[dc]
;W[ef];B[bb];W[bd];B[gb];W[dh];B[hc];W[cm];B[fp];W[gp];B[fo]
;W[eq];B[dq];W[em];B[ch];W[ci];B[dg];W[eh];B[bi];W[bh];B[bg]
;W[cg];B[gq];W[ch];B[hq];W[gn];B[pn];W[go];B[dn];W[dm];B[ip]
;W[ie];B[in];W[ir];B[ep];W[jo];B[io];W[or];B[bn];W[bm];B[qi]
;W[hl];B[nd];W[oh];B[ph];W[oi];B[pj];W[nk];B[il];W[iq];B[hp]
;W[ik];B[hm];W[gl];B[jl];W[jk];B[og];W[qd];B[pc])
)";

static constexpr char kSgfNonStandardSize[] = R"((;
EV[Pro 13x13 tournament]
RO[1]
PB[Mimura Tomoyasu]
BR[9p]
PW[Cho U]
WR[9p]
KM[6.5]
RE[W+R]
DT[2014-08-31]
PC[Nihon Ki-in]
SZ[13]

;B[jd];W[dc];B[kj];W[dk];B[di];W[cg];B[ck];W[cl];B[cj];W[gk]
;B[eg];W[de];B[el];W[dl];B[ek];W[ej];B[dj];W[em];B[fj];W[fl]
;B[ei];W[ik];B[jl];W[ii];B[jh];W[kk];B[jk];W[jj];B[ji];W[ij]
;B[lk];W[ih];B[jg];W[jc];B[kc];W[ic];B[kb];W[ig];B[jf];W[id]
;B[je];W[il];B[kl];W[bl];B[fe];W[fc];B[ed];W[dd];B[ch];W[dg]
;B[bk];W[fk];B[bh];W[bg];B[ag];W[af];B[ah];W[be];B[he];W[jb]
;B[gc];W[gb];B[ec];W[eb];B[fd];W[fb];B[hc];W[hb];B[ja];W[ef]
;B[ff];W[fg];B[gg];W[fh];B[eh];W[gh];B[bi];W[hf];B[ie];W[gf]
;B[al];W[ej];B[gi];W[ia];B[ka];W[gd])
)";

}  // namespace sgf

#endif
